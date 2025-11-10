"""
Delta-neutral opportunity finder using GARCH(1,1) vs Black–Scholes IV.

What it does:
- Fits GARCH(1,1) (Gaussian MLE) to historical log returns (from underlying prices).
- For each option expiry, forecasts average annualized volatility over n trading days.
- Computes Black–Scholes implied volatility (IV) for each option quote.
- Chooses an ATM (≈50-delta) option per expiry to represent a delta-neutral straddle.
- Scores each expiry by expected delta-hedged carry:
    Expected_PnL_total ≈ 0.5 * Gamma_straddle * S^2 * (σ_GARCH^2 − IV^2) * T
  where Gamma_straddle = 2 * Gamma_call(ATM, IV), T in years.

Install:
    pip install numpy scipy

Notes:
- Use trading-day convention D=252. Calendar days are mapped to trading days with 252/365.
- For robust results: use liquid, near-ATM options (avoid penny quotes / tiny vega wings).
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any, Tuple

import numpy as np
from scipy.optimize import minimize

from mt5_connector import MT5Connector

D_TRADING = 252.0
SQRT_2PI = math.sqrt(2.0 * math.pi)


# ------------------------- Normal + Black–Scholes -------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    if sigma <= 0.0 or T <= 0.0:
        F = S * math.exp((r - q) * max(T, 0.0))
        sign = 1.0 if F > K else -1.0
        big = 1e6 * sign
        return big, big
    vol_sqrtT = sigma * math.sqrt(T)
    mu = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    d1 = mu / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return d1, d2

def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True) -> float:
    if T <= 0.0:
        F = S * math.exp((r - q) * T)
        intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
        return intrinsic * math.exp(-r * T)
    if sigma <= 0.0:
        F = S * math.exp((r - q) * T)
        intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
        return intrinsic * math.exp(-r * T)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    if call:
        return S * dq * norm_cdf(d1) - K * dr * norm_cdf(d2)
    else:
        return K * dr * norm_cdf(-d2) - S * dq * norm_cdf(-d1)

def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * math.sqrt(T) * norm_pdf(d1)

def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, call: bool = True) -> float:
    if T <= 0.0:
        F = S * math.exp((r - q) * T)
        if call:
            return 1.0 if F > K else 0.0
        else:
            return -1.0 if F < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    dq = math.exp(-q * T)
    if call:
        return dq * norm_cdf(d1)
    else:
        return -dq * norm_cdf(-d1)

def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))

def implied_vol(
    S: float, K: float, T: float, r: float, q: float, price: float, call: bool = True,
    sigma_init: float = 0.2, tol: float = 1e-8, max_iter: int = 100, lower: float = 1e-6, upper: float = 5.0
) -> Optional[float]:
    # No-arbitrage bounds
    dr, dq = math.exp(-r * T), math.exp(-q * T)
    F = S * math.exp((r - q) * T)
    intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
    lb = intrinsic * dr
    ub = S * dq if call else K * dr
    if price < lb - 1e-12 or price > ub + 1e-12:
        return None

    def f(sig: float) -> float:
        return bs_price(S, K, T, r, q, sig, call) - price

    a, b = lower, upper
    fa, fb = f(a), f(b)
    if fa > 0.0:
        for a_try in [1e-8, 1e-9]:
            fa_try = f(a_try)
            if fa_try <= 0.0:
                a, fa = a_try, fa_try
                break
        else:
            return None
    if fb < 0.0:
        for b_try in [6.0, 8.0, 10.0]:
            fb_try = f(b_try)
            if fb_try >= 0.0:
                b, fb = b_try, fb_try
                break
        else:
            return None

    x = min(max(sigma_init, a), b)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return max(x, 0.0)
        if fx > 0.0:
            b = x
        else:
            a = x
        vega = bs_vega(S, K, T, r, q, x)
        if vega > 1e-12:
            x_new = x - fx / vega
            if (x_new <= a) or (x_new >= b) or not math.isfinite(x_new):
                x = 0.5 * (a + b)
            else:
                x = x_new
        else:
            x = 0.5 * (a + b)
        if (b - a) < max(1e-12, tol * max(1.0, x)):
            return max(0.5 * (a + b), 0.0)
    return max(0.5 * (a + b), 0.0)


# ------------------------- GARCH(1,1) MLE -------------------------

def log_returns(prices: Iterable[float]) -> np.ndarray:
    prices = np.asarray(list(prices), dtype=float)
    prices = prices[prices > 0.0]
    if prices.size < 2:
        return np.array([], dtype=float)
    return np.diff(np.log(prices))

def sample_variance(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size < 2:
        return eps
    m = float(np.mean(x))
    v = float(np.sum((x - m) ** 2) / (x.size - 1))
    return max(v, eps)

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def logit(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1) for logit")
    return math.log(p / (1.0 - p))

def raw_to_params(w_raw: float, a_raw: float, b_raw: float, eps: float = 1e-6) -> Tuple[float, float, float]:
    omega = math.exp(w_raw)
    alpha_cap = 1.0 - eps
    alpha = alpha_cap * sigmoid(a_raw)
    beta_cap = max(1.0 - eps - alpha, eps)
    beta = beta_cap * sigmoid(b_raw)
    return omega, alpha, beta

def params_to_raw(omega: float, alpha: float, beta: float, eps: float = 1e-6) -> Tuple[float, float, float]:
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= (1.0 - eps):
        raise ValueError("Invalid initial GARCH params")
    w_raw = math.log(omega)
    a_raw = logit(alpha / (1.0 - eps))
    b_cap = 1.0 - eps - alpha
    b_raw = logit(beta / b_cap)
    return w_raw, a_raw, b_raw

def garch11_negloglik_from_raw(raw: np.ndarray, rets: np.ndarray, var_init: float, eps: float = 1e-12) -> float:
    w_raw, a_raw, b_raw = float(raw[0]), float(raw[1]), float(raw[2])
    omega, alpha, beta = raw_to_params(w_raw, a_raw, b_raw)
    v = max(var_init, eps)
    neg_ll = 0.0
    for r in rets:
        v = max(v, eps)
        neg_ll += 0.5 * (math.log(2.0 * math.pi) + math.log(v) + (r * r) / v)
        v = omega + alpha * (r * r) + beta * v
    return neg_ll

@dataclass
class Garch11Result:
    omega: float
    alpha: float
    beta: float
    loglik: float
    converged: bool
    message: str
    n_obs: int

def garch11_mle(
    rets: Iterable[float],
    demean: bool = True,
    eps: float = 1e-6,
    init_params: Optional[Tuple[float, float, float]] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
) -> Garch11Result:
    r = np.asarray(list(rets), dtype=float)
    if r.size < 2:
        raise ValueError("Not enough returns for GARCH estimation")
    if demean:
        r = r - np.mean(r)

    var_init = sample_variance(r, eps=1e-12)

    if init_params is None:
        alpha0, beta0 = 0.05, 0.90
        denom = max(1.0 - alpha0 - beta0, 1e-6)
        omega0 = var_init * denom
        init_params = (max(omega0, 1e-12), alpha0, beta0)

    x0 = np.array(params_to_raw(*init_params, eps=eps), dtype=float)

    def obj(x: np.ndarray) -> float:
        return garch11_negloglik_from_raw(x, r, var_init=var_init)

    res = minimize(
        obj, x0, method=method, options={"maxiter": 2000, "disp": False, **(options or {})}
    )
    w_hat, a_hat, b_hat = res.x
    omega, alpha, beta = raw_to_params(w_hat, a_hat, b_hat, eps=eps)
    loglik = -obj(res.x)

    return Garch11Result(
        omega=omega, alpha=alpha, beta=beta,
        loglik=loglik, converged=bool(res.success),
        message=str(res.message), n_obs=r.size
    )


# ------------------------- Forecasting helpers -------------------------

def garch11_forecast_avg_daily_variance(
    last_var: float,
    last_ret2: float,
    omega: float,
    alpha: float,
    beta: float,
    n_steps: int
) -> float:
    lam = alpha + beta
    v_star = omega / (1.0 - lam)
    v_next = omega + alpha * last_ret2 + beta * last_var
    if n_steps <= 0:
        return v_next
    if abs(lam - 1.0) < 1e-12:
        return v_next
    geom_sum = (1.0 - lam**n_steps) / (1.0 - lam)
    avg = v_star + (v_next - v_star) * (geom_sum / n_steps)
    return avg

def annualize_from_avg_daily_variance(avg_daily_var: float, D: float = D_TRADING) -> float:
    return math.sqrt(D * avg_daily_var)

def calendar_to_trading_days(days_calendar: int) -> int:
    return max(1, int(round(days_calendar * (D_TRADING / 252.0))))


# ------------------------- Core analysis -------------------------

@dataclass
class OptionQuote:
    type: str     # "call" or "put"
    K: float
    T_days_calendar: int
    mid: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

@dataclass
class ExpiryScore:
    expiry_days_cal: int
    n_trading_days: int
    K_atm: float
    iv_atm: float
    delta_atm: float
    garch_vol_ann: float
    variance_edge: float
    expected_pnl_total: float
    expected_pnl_per_day: float

def pick_mid(q: OptionQuote) -> Optional[float]:
    if q.mid is not None:
        return q.mid
    if q.bid is not None and q.ask is not None and q.ask >= q.bid and q.bid >= 0.0:
        return 0.5 * (q.bid + q.ask)
    return None

def analyze_delta_neutral_opportunities(
    S: float,
    r: float,
    q: float,
    price_history: Iterable[float],
    chain: List[OptionQuote],
    demean_returns: bool = True,
    delta_target: float = 0.5,
    delta_tolerance: float = 0.2
) -> Tuple[List[ExpiryScore], Optional[ExpiryScore], Optional[ExpiryScore], Garch11Result]:
    """
    - Fits GARCH to price_history.
    - For each expiry, selects an ATM (|delta - 0.5| small) option (prefers calls).
    - Computes IV and GARCH vol alignment and expected delta-hedged carry for an ATM straddle.
    Returns:
        per-expiry scores, best_long (max positive), best_short (most negative), garch_fit
    """
    # GARCH fit
    rets = log_returns(price_history)
    if rets.size < 2:
        raise ValueError("Need at least 2 prices to compute returns")
    garch_fit = garch11_mle(rets, demean=demean_returns)
    last_var = sample_variance(rets)
    last_ret2 = float(rets[-1] ** 2)

    # Group quotes by expiry (calendar days)
    by_expiry: Dict[int, List[OptionQuote]] = {}
    for qte in chain:
        by_expiry.setdefault(qte.T_days_calendar, []).append(qte)

    scores: List[ExpiryScore] = []

    for T_cal, quotes in sorted(by_expiry.items()):
        n_trading = calendar_to_trading_days(T_cal)
        T_years = n_trading / D_TRADING

        # Forecast GARCH average daily variance over horizon and annualize
        avg_daily_var = garch11_forecast_avg_daily_variance(
            last_var, last_ret2, garch_fit.omega, garch_fit.alpha, garch_fit.beta, n_trading
        )
        sigma_garch_ann = annualize_from_avg_daily_variance(avg_daily_var, D=D_TRADING)

        # Compute IV and delta for each quote; keep viable ones
        enriched: List[Tuple[OptionQuote, float, float]] = []  # (quote, iv, delta)
        for qte in quotes:
            mid = pick_mid(qte)
            if mid is None or mid <= 0.0 or qte.K <= 0.0:
                continue
            call_flag = (qte.type.lower() == "call")
            iv = implied_vol(S, qte.K, T_years, r, q, mid, call=call_flag)
            if iv is None or not math.isfinite(iv) or iv <= 0.0:
                continue
            delta = bs_delta(S, qte.K, T_years, r, q, iv, call=call_flag)
            enriched.append((qte, iv, delta))

        if not enriched:
            continue

        # Select ATM proxy: minimal |abs(delta) - target|; prefer calls
        def atm_key(item):
            qte, iv, delta = item
            # For calls, target +0.5; for puts, target -0.5 in absolute terms
            target = delta_target if qte.type.lower() == "call" else -delta_target
            return (abs(delta - target), 0 if qte.type.lower() == "call" else 1)

        enriched.sort(key=atm_key)
        qte_atm, iv_atm, delta_atm = enriched[0]

        # Optional filter to ensure it's reasonably near ATM
        if abs(abs(delta_atm) - delta_target) > delta_tolerance:
            # Too far from ATM; skip this expiry
            continue

        # Compute expected delta-hedged carry of ATM straddle using variance edge
        variance_edge = sigma_garch_ann**2 - iv_atm**2  # positive => long vol edge
        gamma_call = bs_gamma(S, qte_atm.K, T_years, r, q, iv_atm)
        gamma_straddle = 2.0 * gamma_call
        expected_pnl_total = 0.5 * gamma_straddle * (S**2) * variance_edge * T_years
        expected_pnl_per_day = expected_pnl_total / n_trading

        scores.append(ExpiryScore(
            expiry_days_cal=T_cal,
            n_trading_days=n_trading,
            K_atm=qte_atm.K,
            iv_atm=iv_atm,
            delta_atm=delta_atm,
            garch_vol_ann=sigma_garch_ann,
            variance_edge=variance_edge,
            expected_pnl_total=expected_pnl_total,
            expected_pnl_per_day=expected_pnl_per_day
        ))

    if not scores:
        return [], None, None, garch_fit

    # Best opportunities
    best_long = max(scores, key=lambda s: s.expected_pnl_total)  # largest positive
    best_short = min(scores, key=lambda s: s.expected_pnl_total) # most negative

    return scores, best_long, best_short, garch_fit


# ------------------------- Example usage -------------------------

if __name__ == "__main__":
    # Example underlying and rates (replace with your data)
    S = 20.78
    r = 0.14   # continuously compounded
    q = 0.00

    # Example historical adjusted closes (replace with your real price history)
    price_history = [
        20.1, 20.3, 20.0, 20.2, 20.4, 20.1, 20.6, 20.7, 20.5, 20.9, 21.1, 21.0,
        20.8, 20.9, 21.2, 21.0, 20.7, 20.9, 21.3, 21.1, 20.9, 21.0, 20.8, 20.7,
        20.9, 21.0, 20.8, 21.2, 21.0, 20.78
    ]

    mt5_conn = MT5Connector()
    underlying_symbol = "PETR4"
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        exit()

    server_info = mt5_conn.get_account_info().server
    print(f"Connected to MT5 server: {server_info}")  

    spot_price_data = mt5_conn.get_data(underlying_symbol)
    if spot_price_data is None:
        print("Failed to get historical data")
        exit()
    else:
        print(f"Retrieved {spot_price_data.head(1)} for {underlying_symbol}")

   # underlying_symbol_group = underlying_symbol.replace("11", "*")  # Example adjustment
    underlying_symbol_group = underlying_symbol[:4] + "*"  # Generalize to first 4 chars
    print(f"Fetching options chain for underlying group: {underlying_symbol_group}")
    chain_names = mt5_conn.get_options_chain(underlying_symbol_group)
    print(f"Found {len(chain_names)} option symbols")

    chain = [
        # Near 28-day expiry
        OptionQuote(type="call", K=20.5, T_days_calendar=28, bid=0.85, ask=0.95),
        OptionQuote(type="put" , K=20.5, T_days_calendar=28, bid=0.70, ask=0.80),
        OptionQuote(type="call", K=21.0, T_days_calendar=28, bid=0.60, ask=0.70),
        OptionQuote(type="put" , K=21.0, T_days_calendar=28, bid=0.85, ask=0.95),
        # A longer expiry
        OptionQuote(type="call", K=20.5, T_days_calendar=56, bid=1.25, ask=1.40),
        OptionQuote(type="put" , K=20.5, T_days_calendar=56, bid=1.05, ask=1.20),
        OptionQuote(type="call", K=21.0, T_days_calendar=56, bid=1.00, ask=1.15),
        OptionQuote(type="put" , K=21.0, T_days_calendar=56, bid=1.25, ask=1.40),
    ]

    scores, best_long, best_short, fit = analyze_delta_neutral_opportunities(
        S=S, r=r, q=q, price_history=price_history, chain=chain,
        demean_returns=True, delta_target=0.5, delta_tolerance=0.25
    )

    # Print GARCH fit summary
    print("GARCH(1,1) MLE fit:")
    print(f"  converged: {fit.converged}, message: {fit.message}")
    print(f"  omega={fit.omega:.6g}, alpha={fit.alpha:.6g}, beta={fit.beta:.6g}, n_obs={fit.n_obs}")

    # Print per-expiry scores
    if not scores:
        print("\nNo valid ATM opportunities found (check quotes and inputs).")
    else:
        print("\nPer-expiry ATM delta-neutral scores:")
        for s in scores:
            print(
                f"- Expiry ~{s.expiry_days_cal} cal days "
                f"(~{s.n_trading_days} trading): K_atm={s.K_atm:.4f}, "
                f"IV={s.iv_atm:.3%}, GARCH vol={s.garch_vol_ann:.3%}, "
                f"variance edge={s.variance_edge:.6f}, "
                f"expected PnL total≈{s.expected_pnl_total:.4f}, "
                f"per day≈{s.expected_pnl_per_day:.6f}"
            )

        # Print best opportunities
        print("\nBest opportunities (delta-neutral):")
        if best_long and best_long.expected_pnl_total > 0:
            bl = best_long
            print(
                "  Long vol: "
                f"Expiry ~{bl.expiry_days_cal} cal days (~{bl.n_trading_days} trading), "
                f"K_atm={bl.K_atm:.4f}, IV={bl.iv_atm:.3%}, GARCH vol={bl.garch_vol_ann:.3%}, "
                f"variance edge={bl.variance_edge:.6f}, "
                f"expected total carry≈{bl.expected_pnl_total:.4f} "
                f"(per day≈{bl.expected_pnl_per_day:.6f}). "
                "Action: Buy ATM straddle, delta-hedge."
            )
        else:
            print("  Long vol: None with positive expected carry.")

        if best_short and best_short.expected_pnl_total < 0:
            bs = best_short
            print(
                "  Short vol: "
                f"Expiry ~{bs.expiry_days_cal} cal days (~{bs.n_trading_days} trading), "
                f"K_atm={bs.K_atm:.4f}, IV={bs.iv_atm:.3%}, GARCH vol={bs.garch_vol_ann:.3%}, "
                f"variance edge={bs.variance_edge:.6f}, "
                f"expected total carry≈{bs.expected_pnl_total:.4f} "
                f"(per day≈{bs.expected_pnl_per_day:.6f}). "
                "Action: Sell ATM straddle (with risk controls)."
            )
        else:
            print("  Short vol: None with negative expected carry.")

    print("\nNotes:")
    print("- Expected carry uses a simple gamma-theta approximation and assumes IV stays constant (no vega P&L).")
    print("- Use liquid, near-ATM options; avoid penny-priced wings with tiny vega.")
    print("- GARCH forecasts are physical-measure; option IV is risk-neutral and typically higher (variance risk premium).")