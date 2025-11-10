#!/usr/bin/env python3
"""
Compute Gamma Exposure (GEX) for B3 (BVMF) options from MetaTrader 5.

- Underlying: basis field (e.g., BOVA11)
- Option metadata: option_strike, expiration_time, option_right (or derive from series letter)
- OI: session_interest (MT5's open interest; may be 0 if broker doesn’t publish)
- Multiplier: trade_contract_size (often 1.0 BRL per point for B3 equity options)

If OI is not provided (session_interest == 0 for most series), true GEX cannot be computed.
You can optionally substitute a volume proxy, but label it clearly.

Requirements: pip install MetaTrader5 pandas
"""

import math
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def norm_cdf(x: float) -> float:
    # Using error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if right.upper() == "C":
        return df_q * S * norm_cdf(d1) - df_r * K * norm_cdf(d2)
    else:
        return df_r * K * norm_cdf(-d2) - df_q * S * norm_cdf(-d1)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def implied_vol_newton(S, K, T, r, q, right, price, sigma0=0.25, tol=1e-7, max_iter=100) -> Optional[float]:
    # Safeguarded Newton (with basic bounds)
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    # Bounds (vol between 0.0005 and 5)
    lo, hi = 5e-4, 5.0
    sigma = max(lo, min(hi, sigma0))
    for _ in range(max_iter):
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        df_r = math.exp(-r * T)
        df_q = math.exp(-q * T)
        if right.upper() == "C":
            model = df_q * S * norm_cdf(d1) - df_r * K * norm_cdf(d2)
            vega = df_q * S * math.sqrt(T) * norm_pdf(d1)
        else:
            model = df_r * K * norm_cdf(-d2) - df_q * S * norm_cdf(-d1)
            vega = df_q * S * math.sqrt(T) * norm_pdf(d1)
        diff = model - price
        if abs(diff) < tol:
            return max(lo, min(hi, sigma))
        if vega <= 1e-8:
            break
        step = diff / vega
        sigma -= step
        if sigma <= lo or sigma >= hi:
            # Bisection fallback
            sigma = 0.5 * (lo + hi)
        else:
            # Narrow the bracket using sign of diff
            if diff > 0:
                hi = sigma
            else:
                lo = sigma
    return None


def b3_series_letter_to_right_month(letter: str) -> Tuple[Optional[str], Optional[int]]:
    """
    B3 convention (like US OCC):
      Calls: A–L = Jan–Dec
      Puts:  M–X = Jan–Dec
    Returns (right, month) where right in {'C','P'}.
    """
    if not letter or len(letter) != 1 or not letter.isalpha():
        return None, None
    l = letter.upper()
    calls = "ABCDEFGHIJKL"
    puts = "MNOPQRSTUVWX"
    if l in calls:
        month = calls.index(l) + 1
        return "C", month
    if l in puts:
        month = puts.index(l) + 1
        return "P", month
    return None, None


def infer_right_from_symbol_name(name: str, basis: str) -> Optional[str]:
    """
    Try to infer 'C' or 'P' from the series letter immediately after the basis root.
    Example: BOVA[L]118W2 -> 'L' => Call (Dec)
    """
    if not name or not basis:
        return None
    root = basis.split()[0]  # basis is usually like 'BOVA11'
    # Many B3 option codes use the underlying root (without '11') + series letter.
    # We'll try to find the first letter after the underlying base 'BOVA'
    # If basis ends with digits (e.g., BOVA11), strip trailing digits for root match.
    stripped = ''.join([c for c in root if not c.isdigit()])
    idx = name.upper().find(stripped.upper())
    if idx == -1:
        # Fallback: find first alpha block after first 4 letters
        idx = 4
    # Scan for first letter after idx
    for j in range(idx, len(name)):
        if name[j].isalpha():
            right, _ = b3_series_letter_to_right_month(name[j])
            return right
    return None


def main(
    underlying_basis: str = "BOVA11",
    symbol_glob: str = "BOVA*",
    risk_free_rate: float = 0.12,   # annualized (e.g., SELIC ~12% -> 0.12)
    div_yield: float = 0.00,        # set dividend/borrow yield if you have it
    dealer_signed: bool = True
):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    try:
        # Underlying spot (mid)
        u_info = mt5.symbol_info(underlying_basis)
        if not u_info:
            raise RuntimeError(f"Underlying {underlying_basis} not found in MT5.")
        u_tick = mt5.symbol_info_tick(underlying_basis)
        if not u_tick:
            raise RuntimeError(f"No tick for {underlying_basis}.")
        S = (u_tick.bid + u_tick.ask) / 2.0 if u_tick.ask > 0 else u_tick.bid
        if S <= 0:
            raise RuntimeError("Invalid underlying price.")

        now_utc = datetime.now(timezone.utc)
        now_epoch = int(now_utc.timestamp())

        # Gather option symbols for this underlying
        syms = mt5.symbols_get(symbol_glob)
        rows = []
        for s in syms:
            info = mt5.symbol_info(s.name)
            print(f"Processing symbol: {s.name}")
            if not info or info.basis != underlying_basis:
                continue
            if info.option_strike <= 0 or info.expiration_time <= 0:
                continue
            # Determine right
            right = None
            if getattr(info, "option_right", 0) == 1:
                right = "C"
            elif getattr(info, "option_right", 0) == 2:
                right = "P"
            else:
                right = infer_right_from_symbol_name(info.name, info.basis)
            if right not in ("C", "P"):
                continue

            # Mid price
            bid, ask = info.bid, info.ask
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
            # If MT5 provides theoretical price, you can fall back to that:
            theo = getattr(info, "price_theoretical", 0.0) or 0.0
            if mid <= 0 and theo > 0:
                mid = float(theo)

            # Time to expiry (years)
            T = max(0.0, info.expiration_time - now_epoch) / (365.0 * 24 * 3600)

            # OI (session_interest)
            oi = float(getattr(info, "session_interest", 0.0) or 0.0)

            rows.append({
                "symbol": info.name,
                "right": right,
                "strike": float(info.option_strike),
                "expiry_epoch": int(info.expiration_time),
                "T": T,
                "mid": mid,
                "oi": oi,
                "mult": float(info.trade_contract_size or 1.0)
            })

        df = pd.DataFrame(rows)
        if df.empty:
            print("No option series found for basis", underlying_basis)
            return

        # Solve implied vol and gamma; compute dollar gamma per contract
        sigmas = []
        gammas = []
        dollar_gamma = []
        for r in df.itertuples(index=False):
            # Skip if T or price invalid; gamma requires sigma
            sigma = None
            if r.mid > 0 and r.T > 0:
                sigma = implied_vol_newton(S, r.strike, r.T, risk_free_rate, div_yield, r.right, r.mid, sigma0=0.25)
            if not sigma:
                sigmas.append(float("nan"))
                gammas.append(0.0)
                dollar_gamma.append(0.0)
                continue
            g = bs_gamma(S, r.strike, r.T, risk_free_rate, div_yield, sigma)
            dg = (S ** 2) * g * r.mult
            sigmas.append(sigma)
            gammas.append(g)
            dollar_gamma.append(dg)

        df["iv"] = sigmas
        df["gamma"] = gammas
        df["dollar_gamma_per_contract"] = dollar_gamma

        # Signed exposure
        sign = df["right"].map({"C": +1.0, "P": -1.0})
        df["gex_line"] = df["dollar_gamma_per_contract"] * df["oi"] * sign

        total_gex = float(df["gex_line"].sum())
        total_gex_dealer = -total_gex if dealer_signed else total_gex

        # Simple profiles
        by_strike = df.groupby("strike", as_index=False)["gex_line"].sum().rename(columns={"gex_line": "gex_strike"})
        by_expiry = df.groupby("expiry_epoch", as_index=False)["gex_line"].sum().rename(columns={"gex_line": "gex_expiry"})
        by_expiry["expiry_utc"] = by_expiry["expiry_epoch"].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).strftime("%Y-%m-%d"))

        # Report
        zero_oi_ratio = (df["oi"] <= 0).mean()
        if zero_oi_ratio > 0.5:
            print(f"WARNING: {zero_oi_ratio:.0%} of series have OI=0 (session_interest). Broker may not publish OI; true GEX will be unreliable.")
        print(f"Underlying {underlying_basis} mid: {S:.4f}  (r={risk_free_rate:.2%}, q={div_yield:.2%})")
        print(f"Total GEX (calls - puts): {total_gex:,.0f} BRL-gamma")
        print(f"Dealer-signed GEX:        {total_gex_dealer:,.0f} BRL-gamma")

        print("\nGEX by strike (top 10 by magnitude):")
        print(by_strike.reindex(by_strike.gex_strike.abs().sort_values(ascending=False).index).head(10).to_string(index=False))

        print("\nGEX by expiry:")
        print(by_expiry.sort_values("expiry_epoch")[["expiry_utc","gex_expiry"]].to_string(index=False))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    # Tune inputs if needed
    main(
        underlying_basis="BOVA11",
        symbol_glob="BOVA*",
        risk_free_rate=0.12,  # adjust to current SELIC/DI
        div_yield=0.00,
        dealer_signed=True
    )