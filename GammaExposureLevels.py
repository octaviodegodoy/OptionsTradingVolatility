# -*- coding: utf-8 -*-
"""
Options GEX Analytics — Outspoken Market version
-------------------------------------------------
Supports two modes:
  BOVA11 — B3 Brazilian options via COTAHIST + OI proxy
  SPY    — US equity options via yfinance (real OI) for cross-checking
           against public GEX services (SpotGamma, SqueezeMetrics, etc.)

Performs:
- Global and range-based Put/Call Ratio
- IV skew (OTM puts vs OTM calls)
- Notional by strike (volume financeiro)
- Gamma Exposure (Customer/Dealer)
- Call/Put walls and Gamma Flip
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta

from constants import ASSET_SYMBOL

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# ============================================================
# MODE SELECTION — set to "BOVA11" or "SPY"
# ============================================================
VALIDATION_MODE = "BOVA11"   # "BOVA11" for Brazilian market, "SPY" for US cross-check

# Mode-dependent imports and constants
if VALIDATION_MODE == "BOVA11":
    from mt5_connector import MT5Connector
    from get_b3_data import fetch_b3_historical_file, fetch_open_interest
    UNDERLYING = "BOVA11"
    RISK_FREE_RATE = 0.1425       # Brazilian SELIC
    CONTRACT_MULTIPLIER = 1       # B3 mini options = 1 share
else:
    import yfinance as yf
    UNDERLYING = "SPY"
    RISK_FREE_RATE = 0.045        # US Fed Funds / T-Bill proxy
    CONTRACT_MULTIPLIER = 100     # US equity options = 100 shares

# Path to external OI CSV file (set to None to use multi-day volume proxy)
# Expected CSV columns: ticker, oi  (optional: strike, type, expiration)
OI_CSV_PATH = None  # e.g., os.path.join(SCRIPT_DIR, "bova11_oi.csv")


# ============================================================
# Black-Scholes Pricing & Greeks
# ============================================================
def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes Gamma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes Delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if option_type == 'call' else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1


def implied_vol(price, S, K, T, r, option_type='call'):
    """Solve for implied volatility using Brent's method."""
    if T <= 0 or price <= 0 or K <= 0:
        return 0.30
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if price <= intrinsic + 1e-8:
        return 0.30
    try:
        return brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price,
            0.01, 5.0, xtol=1e-6, maxiter=100
        )
    except (ValueError, RuntimeError):
        return 0.30


def find_gamma_flip(strikes, gex_values, spot, window=3):
    """
    Find the gamma flip point — the zero crossing of smoothed GEX nearest to spot.
    Uses linear interpolation between sign-change points for precision.
    Falls back to NaN if no meaningful crossing exists.
    """
    if len(strikes) < 4:
        return np.nan
    smooth = pd.Series(gex_values).rolling(window, center=True, min_periods=1).mean().values
    # Find all sign-change crossings via linear interpolation
    crossings = []
    for i in range(1, len(smooth)):
        if smooth[i - 1] * smooth[i] < 0:  # sign change
            s0, s1 = strikes[i - 1], strikes[i]
            g0, g1 = smooth[i - 1], smooth[i]
            cross = s0 + (0 - g0) * (s1 - s0) / (g1 - g0)
            crossings.append(cross)
    if crossings:
        # Return the crossing closest to spot
        return min(crossings, key=lambda c: abs(c - spot))
    # Fallback: argmin(|smooth|) but only within ±30% of spot to avoid extreme strikes
    low_bound = spot * 0.70
    high_bound = spot * 1.30
    mask = (strikes >= low_bound) & (strikes <= high_bound)
    if mask.any():
        local_idx = np.where(mask)[0]
        best = local_idx[np.argmin(np.abs(smooth[local_idx]))]
        return strikes[best]
    return np.nan


# ============================================================
# B3 Data Loader with Greek Computation
# ============================================================
def load_b3_options_data(underlying, spot, date=None):
    """
    Fetch options from B3 historical file and compute Greeks via Black-Scholes.
    Returns DataFrame with columns:
        Ticker, Tipo, Strike, Ultimo, IV, Delta, Gamma, Theta ($), Vega, Tit., Lanç., VolFin
    """
    raw = fetch_b3_historical_file(date)
    if raw.empty:
        return pd.DataFrame()

    prefix = underlying[:4].upper()
    call_letters = set('ABCDEFGHIJKL')
    put_letters = set('MNOPQRSTUVWX')

    options = raw[raw['ticker'].str.startswith(prefix, na=False)].copy()
    if options.empty:
        print(f"No options found for {underlying}")
        return pd.DataFrame()

    # Classify call/put from the series letter (5th char of ticker)
    def classify_type(ticker):
        if len(ticker) > 4:
            letter = ticker[4].upper()
            if letter in call_letters:
                return 'CALL'
            elif letter in put_letters:
                return 'PUT'
        return None

    options['Tipo'] = options['ticker'].apply(classify_type)
    options = options.dropna(subset=['Tipo'])

    # Parse expiration and compute time to expiry + DTE
    now = datetime.now()
    def parse_expiration(exp_str):
        try:
            exp_date = datetime.strptime(str(exp_str).strip(), '%Y%m%d')
            dte = max((exp_date - now).days, 0)
            T = max(dte / 365.0, 1 / 365)
            return T, dte, exp_date
        except (ValueError, TypeError):
            return 28 / 365, 28, now  # fallback ~1 month

    parsed = options['expiration'].apply(parse_expiration)
    options['T'] = parsed.apply(lambda x: x[0])
    options['DTE'] = parsed.apply(lambda x: x[1])
    options['Expiration'] = parsed.apply(lambda x: x[2])

    r = RISK_FREE_RATE
    ivs, gammas, deltas = [], [], []
    for _, row in options.iterrows():
        opt_type = row['Tipo'].lower()
        strike = float(row['strike'])
        close = float(row['close'])
        T = float(row['T'])

        # Implied vol from market price
        if close > 0 and strike > 0:
            iv = implied_vol(close, spot, strike, T, r, opt_type)
        else:
            iv = 0.30

        gammas.append(bs_gamma(spot, strike, T, r, iv))
        deltas.append(bs_delta(spot, strike, T, r, iv, opt_type))
        ivs.append(iv)  # stored as decimal

    df = pd.DataFrame({
        'Ticker': options['ticker'].values,
        'Tipo': options['Tipo'].values,
        'Strike': options['strike'].values.astype(float),
        'Ultimo': options['close'].values.astype(float),
        'IV': np.array(ivs),
        'Delta': np.array(deltas),
        'Gamma': np.array(gammas),
        'Theta ($)': 0.0,
        'Vega': 0.0,
        'Tit.': options['quantity'].values.astype(float),  # daily volume (fallback)
        'Lanc.': 0.0,
        'VolFin': options['volume'].values.astype(float),
        'DTE': options['DTE'].values.astype(int),
        'Expiration': options['Expiration'].values,
    })

    # ---- Merge real OI data if available ----
    oi_source = 'daily_volume'
    try:
        oi_data = fetch_open_interest(
            underlying=underlying,
            oi_csv_path=OI_CSV_PATH,
            multiday_days=5,
        )
        if not oi_data.empty and 'ticker' in oi_data.columns and 'oi' in oi_data.columns:
            oi_map = oi_data.set_index('ticker')['oi'].to_dict()
            oi_source = oi_data['oi_source'].iloc[0] if 'oi_source' in oi_data.columns else 'external'

            # Replace Tit. with real OI where available
            matched = 0
            for idx, row in df.iterrows():
                ticker = row['Ticker']
                if ticker in oi_map and oi_map[ticker] > 0:
                    df.at[idx, 'Tit.'] = oi_map[ticker]
                    matched += 1

            print(f"[*] OI source: {oi_source} | Matched {matched}/{len(df)} options")
        else:
            print("[*] OI source: daily_volume (no better source available)")
    except Exception as e:
        print(f"[*] OI source: daily_volume (fetch error: {e})")

    calls_n = (df['Tipo'] == 'CALL').sum()
    puts_n = (df['Tipo'] == 'PUT').sum()
    dte_dist = df['DTE'].value_counts().sort_index()
    print(f"[*] Built chain: {len(df)} records ({calls_n} calls, {puts_n} puts)")
    print(f"   DTE distribution: {dict(dte_dist.head(5))}")
    print(f"   GEX weight: {'OI' if 'volume' not in oi_source else 'Volume (OI proxy)'}")
    return df


# ============================================================
# SPY Data Loader via yfinance (Validation Mode)
# ============================================================
def _yahoo_direct_options(symbol="SPY"):
    """
    Fallback: fetch options data directly from Yahoo Finance query API
    with crumb authentication. Returns (session, crumb, base_url).
    """
    import requests
    session = requests.Session()
    session.headers['User-Agent'] = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    # Step 1: Get cookie
    session.get('https://fc.yahoo.com', timeout=10, allow_redirects=True)
    # Step 2: Get crumb
    crumb = session.get(
        'https://query2.finance.yahoo.com/v1/test/getcrumb', timeout=10
    ).text
    base_url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
    return session, crumb, base_url


def load_spy_options_data(spot=None, max_expirations=6, retry_attempts=5, retry_delay=10):
    """
    Fetch SPY options chain with real open interest.
    Primary: direct Yahoo Finance query API with crumb auth.
    Fallback: yfinance library.
    Returns (DataFrame, spot_price).
    """
    print("=" * 75)
    print("SPY VALIDATION MODE — Loading options via Yahoo Finance")
    print("=" * 75)

    records = []
    today = datetime.now()

    # --- Step 1: Establish authenticated session ---
    session = None
    crumb = None
    base_url = None
    for attempt in range(retry_attempts):
        try:
            session, crumb, base_url = _yahoo_direct_options("SPY")
            print(f"[*] Yahoo session established (crumb obtained)")
            break
        except Exception as e:
            print(f"[!] Session setup attempt {attempt+1}: {e}")
            time.sleep(retry_delay)

    # --- Step 2: First API call to get spot + expirations + first chain ---
    first_data = None
    if session and crumb:
        for attempt in range(retry_attempts):
            try:
                resp = session.get(f"{base_url}?crumb={crumb}", timeout=15)
                if resp.status_code == 429:
                    print(f"[!] Rate limited (attempt {attempt+1}), waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                resp.raise_for_status()
                first_data = resp.json()
                break
            except Exception as e:
                print(f"[!] Direct API attempt {attempt+1}: {e}")
                time.sleep(retry_delay)

    if first_data is None:
        # Try yfinance as fallback
        print("[!] Direct API failed, trying yfinance...")
        try:
            ticker = yf.Ticker("SPY")
            if spot is None:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    spot = float(hist['Close'].iloc[-1])
            expirations = list(ticker.options[:max_expirations])
            for exp_str in expirations:
                time.sleep(2)
                chain = ticker.option_chain(exp_str)
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = max((exp_date - today).days, 1)
                T = dte / 365.0
                for opt_type, df_raw in [("CALL", chain.calls), ("PUT", chain.puts)]:
                    for _, row in df_raw.iterrows():
                        strike = float(row['strike'])
                        oi = int(row.get('openInterest', 0) or 0) if not pd.isna(row.get('openInterest')) else 0
                        volume = int(row.get('volume', 0) or 0) if not pd.isna(row.get('volume')) else 0
                        last_price = float(row.get('lastPrice', 0) or 0) if not pd.isna(row.get('lastPrice')) else 0.0
                        yf_iv = float(row.get('impliedVolatility', 0) or 0) if not pd.isna(row.get('impliedVolatility')) else 0.0
                        if oi == 0: continue
                        moneyness = strike / spot
                        if moneyness < 0.80 or moneyness > 1.20: continue
                        iv = yf_iv if yf_iv > 0.001 else implied_vol(spot, strike, T, RISK_FREE_RATE, last_price, opt_type.lower())
                        gamma = bs_gamma(spot, strike, T, RISK_FREE_RATE, iv)
                        delta = bs_delta(spot, strike, T, RISK_FREE_RATE, iv, opt_type.lower())
                        records.append({
                            'Ticker': f"SPY_{exp_str}_{strike:.0f}_{opt_type[0]}",
                            'Tipo': opt_type, 'Strike': strike, 'Ultimo': last_price,
                            'IV': iv, 'Delta': delta, 'Gamma': gamma,
                            'Theta ($)': 0.0, 'Vega': 0.0, 'Tit.': oi, 'Lanc.': 0,
                            'VolFin': last_price * oi * CONTRACT_MULTIPLIER,
                            'DTE': dte, 'Expiration': exp_date.strftime('%Y-%m-%d'), 'Volume': volume,
                        })
                print(f"  [OK] {exp_str}")
        except Exception as e:
            print(f"[X] yfinance fallback also failed: {e}")
            return pd.DataFrame(), spot

        if not records:
            return pd.DataFrame(), spot
        df = pd.DataFrame(records)
        return df, spot

    # --- Parse direct API response ---
    result = first_data['optionChain']['result'][0]
    spot = spot or result['quote'].get('regularMarketPrice',
                                        result['quote'].get('regularMarketPreviousClose', 0))
    print(f"[*] SPY spot: ${spot:.2f}")

    exp_timestamps = result.get('expirationDates', [])
    if not exp_timestamps:
        print("[X] No expirations found")
        return pd.DataFrame(), spot

    # Limit expirations
    exp_timestamps = exp_timestamps[:max_expirations]
    print(f"[*] Loading {len(exp_timestamps)} expirations...")

    def _parse_chain(options_data, exp_ts):
        """Parse one expiration's options from Yahoo JSON."""
        exp_date = datetime.fromtimestamp(exp_ts)
        exp_str = exp_date.strftime('%Y-%m-%d')
        dte = max((exp_date - today).days, 1)
        T = dte / 365.0

        for opt_type_key, opt_type_label in [('calls', 'CALL'), ('puts', 'PUT')]:
            for opt in options_data.get(opt_type_key, []):
                strike = opt.get('strike', 0)
                oi = opt.get('openInterest', 0) or 0
                volume = opt.get('volume', 0) or 0
                last_price = opt.get('lastPrice', 0) or 0
                yf_iv = opt.get('impliedVolatility', 0) or 0

                if oi == 0:
                    continue
                moneyness = strike / spot if spot > 0 else 0
                if moneyness < 0.80 or moneyness > 1.20:
                    continue

                iv = yf_iv if yf_iv > 0.001 else implied_vol(
                    spot, strike, T, RISK_FREE_RATE, last_price, opt_type_label.lower())
                gamma = bs_gamma(spot, strike, T, RISK_FREE_RATE, iv)
                delta = bs_delta(spot, strike, T, RISK_FREE_RATE, iv, opt_type_label.lower())
                notional = last_price * oi * CONTRACT_MULTIPLIER

                records.append({
                    'Ticker': f"SPY_{exp_str}_{strike:.0f}_{opt_type_label[0]}",
                    'Tipo': opt_type_label,
                    'Strike': strike,
                    'Ultimo': last_price,
                    'IV': iv,
                    'Delta': delta,
                    'Gamma': gamma,
                    'Theta ($)': 0.0,
                    'Vega': 0.0,
                    'Tit.': oi,
                    'Lanc.': 0,
                    'VolFin': notional,
                    'DTE': dte,
                    'Expiration': exp_str,
                    'Volume': volume,
                })
        call_count = sum(1 for r in records if r.get('Expiration') == exp_str and r['Tipo'] == 'CALL')
        put_count = sum(1 for r in records if r.get('Expiration') == exp_str and r['Tipo'] == 'PUT')
        print(f"  [OK] {exp_str} (DTE {dte}): {call_count} calls, {put_count} puts")

    # Parse first expiration (already in the response)
    if result.get('options'):
        first_opt = result['options'][0]
        first_ts = first_opt.get('expirationDate', exp_timestamps[0])
        _parse_chain(first_opt, first_ts)

    # Fetch remaining expirations
    for exp_ts in exp_timestamps[1:]:
        time.sleep(0.5)  # polite delay
        for attempt in range(retry_attempts):
            try:
                resp = session.get(f"{base_url}?date={exp_ts}&crumb={crumb}", timeout=15)
                if resp.status_code == 429:
                    print(f"  [!] Rate limited on exp {exp_ts}, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                opts = data['optionChain']['result'][0].get('options', [])
                if opts:
                    _parse_chain(opts[0], exp_ts)
                break
            except Exception as e:
                print(f"  [!] Expiration {exp_ts} attempt {attempt+1}: {e}")
                time.sleep(retry_delay)

    if not records:
        print("[X] No SPY options data retrieved")
        return pd.DataFrame(), spot

    df = pd.DataFrame(records)
    calls_n = (df['Tipo'] == 'CALL').sum()
    puts_n = (df['Tipo'] == 'PUT').sum()
    total_oi = df['Tit.'].sum()
    print(f"\n[*] SPY chain: {len(df)} records ({calls_n} calls, {puts_n} puts)")
    print(f"   Total OI: {total_oi:,.0f} contracts")
    print(f"   GEX weight: Real Open Interest (Yahoo Finance)")
    return df, spot


# ============================================================
# SPY GEX Validation Analysis
# ============================================================
async def analyze_spy_validation():
    """
    Run GEX analysis on SPY and print results in a format comparable
    to public GEX services (SpotGamma, SqueezeMetrics, Unusual Whales).
    """
    print("\n" + "=" * 75)
    print("      SPY GEX VALIDATION — Cross-Check vs Public Services")
    print("=" * 75)

    result = load_spy_options_data()
    if isinstance(result, tuple):
        df, spot = result
    else:
        df, spot = result, None

    if df.empty or spot is None:
        print("[X] No data — cannot run validation")
        return

    # --- Put/Call Ratio ---
    call_oi = df.loc[df['Tipo'] == 'CALL', 'Tit.'].sum()
    put_oi = df.loc[df['Tipo'] == 'PUT', 'Tit.'].sum()
    pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan
    print(f"\n--- Put/Call Ratio (OI) ---")
    print(f"  Call OI: {call_oi:>12,.0f}")
    print(f"  Put OI:  {put_oi:>12,.0f}")
    print(f"  PCR:     {pcr_oi:>12.3f}")

    call_vol = df.loc[df['Tipo'] == 'CALL', 'Volume'].sum()
    put_vol = df.loc[df['Tipo'] == 'PUT', 'Volume'].sum()
    pcr_vol = put_vol / call_vol if call_vol > 0 else np.nan
    print(f"\n--- Put/Call Ratio (Volume) ---")
    print(f"  Call Vol: {call_vol:>11,.0f}")
    print(f"  Put Vol:  {put_vol:>11,.0f}")
    print(f"  PCR:      {pcr_vol:>11.3f}")

    # --- IV Skew ---
    otm_puts = df[(df['Tipo'] == 'PUT') & (df['Strike'] < spot)]
    otm_calls = df[(df['Tipo'] == 'CALL') & (df['Strike'] > spot)]
    avg_put_iv = otm_puts['IV'].mean() if not otm_puts.empty else np.nan
    avg_call_iv = otm_calls['IV'].mean() if not otm_calls.empty else np.nan
    skew = avg_put_iv - avg_call_iv if not (np.isnan(avg_put_iv) or np.isnan(avg_call_iv)) else np.nan
    print(f"\n--- IV Skew ---")
    print(f"  Avg OTM Put IV:  {avg_put_iv*100:>6.2f}%")
    print(f"  Avg OTM Call IV: {avg_call_iv*100:>6.2f}%")
    print(f"  Skew (P-C):      {skew*100:>+6.2f}%")

    # --- GEX Computation ---
    # Same formula as BOVA11: GEX = Gamma * S^2 * OI * multiplier * sign
    # sign: +1 for calls (dealer long gamma), -1 for puts (dealer short gamma)
    df['sign'] = df['Tipo'].map({'CALL': 1, 'PUT': -1})
    df['GEX'] = (df['Gamma'] * spot ** 2 * df['Tit.']
                 * CONTRACT_MULTIPLIER * df['sign'] * 0.01)

    total_gex = df['GEX'].sum()
    total_gex_mil = total_gex / 1e6
    print(f"\n--- Total GEX ---")
    print(f"  Total GEX: ${total_gex_mil:>+,.2f}M")
    print(f"  Gamma Environment: {'POSITIVE (support/pin)' if total_gex > 0 else 'NEGATIVE (volatility/acceleration)'}")

    # --- GEX by Strike ---
    gex_by_strike = df.groupby('Strike')['GEX'].sum().sort_index()
    strikes = gex_by_strike.index.values
    gex_vals = gex_by_strike.values

    # --- Call Wall (highest call GEX above spot) ---
    call_gex = df[df['Tipo'] == 'CALL'].groupby('Strike')['GEX'].sum()
    call_above = call_gex[call_gex.index >= spot]
    call_wall = call_above.idxmax() if not call_above.empty else np.nan

    # --- Put Wall (most negative put GEX below spot) ---
    put_gex = df[df['Tipo'] == 'PUT'].groupby('Strike')['GEX'].sum()
    put_below = put_gex[put_gex.index <= spot]
    put_wall = put_below.idxmin() if not put_below.empty else np.nan

    # --- Gamma Flip ---
    gamma_flip = find_gamma_flip(strikes, gex_vals, spot)

    # --- Key Levels (SpotGamma-style output) ---
    print(f"\n{'='*75}")
    print(f"   SPY KEY GEX LEVELS (compare with SpotGamma / SqueezeMetrics)")
    print(f"{'='*75}")
    print(f"  Spot Price:   ${spot:>10,.2f}")
    print(f"  Call Wall:    ${call_wall:>10,.2f}  (max dealer long gamma)")
    print(f"  Put Wall:     ${put_wall:>10,.2f}  (max dealer short gamma)")
    print(f"  Gamma Flip:   ${gamma_flip:>10,.2f}  (positive->negative gamma transition)")
    print(f"  Total GEX:    ${total_gex_mil:>+10,.2f}M")

    # Validate ordering
    ordering_ok = put_wall <= spot <= call_wall
    flip_ok = put_wall <= gamma_flip <= call_wall if not np.isnan(gamma_flip) else False
    print(f"\n  Ordering Check:")
    print(f"    Put Wall <= Spot <= Call Wall:  {'PASS' if ordering_ok else 'FAIL'}")
    print(f"    Put Wall <= Flip <= Call Wall:  {'PASS' if flip_ok else 'WARN'}")

    # --- Regime Classification ---
    if total_gex > 0 and spot > gamma_flip:
        regime = "POSITIVE GAMMA — Low vol, mean-reverting, pinning near call wall"
    elif total_gex > 0 and spot <= gamma_flip:
        regime = "TRANSITION — Positive GEX but below flip, watch for breakdown"
    elif total_gex < 0 and spot < gamma_flip:
        regime = "NEGATIVE GAMMA — High vol, trend-following, acceleration moves"
    else:
        regime = "TRANSITION — Negative GEX but above flip, watch for stabilisation"
    print(f"\n  Market Regime: {regime}")

    # --- Top Strikes by GEX ---
    print(f"\n--- Top 10 Strikes by |GEX| ---")
    top_strikes = gex_by_strike.abs().nlargest(10).index
    for s in sorted(top_strikes):
        gex_m = gex_by_strike[s] / 1e6
        label = "CALL dom" if gex_by_strike[s] > 0 else "PUT dom"
        print(f"  ${s:>8.0f}  |  GEX: ${gex_m:>+8.2f}M  |  {label}")

    # --- Next Friday Focus ---
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0 and today.hour >= 16:
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    nf_str = next_friday.strftime('%Y-%m-%d')
    df_nf = df[df['Expiration'] == nf_str]
    if not df_nf.empty:
        nf_gex = df_nf.groupby('Strike')['GEX'].sum()
        nf_total = nf_gex.sum() / 1e6
        nf_call_wall = nf_gex[nf_gex.index >= spot].idxmax() if not nf_gex[nf_gex.index >= spot].empty else np.nan
        nf_put_wall = nf_gex[nf_gex.index <= spot].idxmin() if not nf_gex[nf_gex.index <= spot].empty else np.nan
        print(f"\n--- Next Friday ({nf_str}) GEX ---")
        print(f"  Call Wall: ${nf_call_wall:>8,.0f}")
        print(f"  Put Wall:  ${nf_put_wall:>8,.0f}")
        print(f"  Total GEX: ${nf_total:>+8.2f}M")
    else:
        # Try nearest expiration
        all_exps = sorted(df['Expiration'].unique())
        if all_exps:
            nearest = all_exps[0]
            df_near = df[df['Expiration'] == nearest]
            near_gex = df_near.groupby('Strike')['GEX'].sum()
            print(f"\n--- Nearest Expiration ({nearest}) GEX ---")
            print(f"  Total: ${near_gex.sum()/1e6:>+8.2f}M")

    # === CHARTS ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"SPY GEX Validation — Spot: ${spot:.2f}", fontsize=14, fontweight='bold')

    # --- Chart 1: GEX by Strike ---
    ax = axes[0]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in gex_vals / 1e6]
    ax.bar(strikes, gex_vals / 1e6, width=(strikes[1] - strikes[0]) * 0.7 if len(strikes) > 1 else 1,
           color=colors, alpha=0.8, edgecolor='white', linewidth=0.3)

    # Smooth line
    if len(strikes) >= 5:
        smooth = pd.Series(gex_vals / 1e6).rolling(5, center=True, min_periods=1).mean()
        ax.plot(strikes, smooth, color='navy', linewidth=1.5, alpha=0.7, label='Smoothed GEX')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(spot, color='blue', linewidth=2, linestyle='--', label=f'Spot ${spot:.0f}')
    if not np.isnan(call_wall):
        ax.axvline(call_wall, color='green', linewidth=1.5, linestyle=':', label=f'Call Wall ${call_wall:.0f}')
    if not np.isnan(put_wall):
        ax.axvline(put_wall, color='red', linewidth=1.5, linestyle=':', label=f'Put Wall ${put_wall:.0f}')
    if not np.isnan(gamma_flip):
        ax.axvline(gamma_flip, color='orange', linewidth=1.5, linestyle='-.', label=f'Flip ${gamma_flip:.1f}')

    ax.set_xlabel('Strike')
    ax.set_ylabel('GEX ($M)')
    ax.set_title('Gamma Exposure by Strike (All Expirations)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Shade positive/negative zones
    ax.fill_between(strikes, 0, gex_vals / 1e6,
                     where=gex_vals > 0, color='green', alpha=0.05)
    ax.fill_between(strikes, 0, gex_vals / 1e6,
                     where=gex_vals < 0, color='red', alpha=0.05)

    # Focus x-axis around spot ±10%
    ax.set_xlim(spot * 0.92, spot * 1.08)

    # --- Chart 2: OI by Strike (calls vs puts) ---
    ax2 = axes[1]
    call_oi_by_strike = df[df['Tipo'] == 'CALL'].groupby('Strike')['Tit.'].sum()
    put_oi_by_strike = df[df['Tipo'] == 'PUT'].groupby('Strike')['Tit.'].sum()

    bar_w = (strikes[1] - strikes[0]) * 0.35 if len(strikes) > 1 else 0.5
    if not call_oi_by_strike.empty:
        ax2.bar(call_oi_by_strike.index - bar_w / 2, call_oi_by_strike.values / 1000,
                width=bar_w, color='#2ecc71', alpha=0.7, label='Call OI')
    if not put_oi_by_strike.empty:
        ax2.bar(put_oi_by_strike.index + bar_w / 2, put_oi_by_strike.values / 1000,
                width=bar_w, color='#e74c3c', alpha=0.7, label='Put OI')

    ax2.axvline(spot, color='blue', linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('OI (thousands)')
    ax2.set_title('Open Interest by Strike')
    ax2.legend(fontsize=8)
    ax2.set_xlim(spot * 0.92, spot * 1.08)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "spy_gex_validation.png"), dpi=150, bbox_inches='tight')
    print(f"\n[*] Chart saved: spy_gex_validation.png")
    plt.show()

    # --- Reference comparison template ---
    print(f"\n{'='*75}")
    print("  CROSS-CHECK TEMPLATE (compare with SpotGamma / SqueezeMetrics)")
    print(f"{'='*75}")
    print(f"  {'Metric':<20} {'Our Model':>15} {'SpotGamma':>15} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Call Wall':<20} {'$'+f'{call_wall:,.0f}':>15} {'(enter)':>15} {'':>10}")
    print(f"  {'Put Wall':<20} {'$'+f'{put_wall:,.0f}':>15} {'(enter)':>15} {'':>10}")
    print(f"  {'Gamma Flip':<20} {'$'+f'{gamma_flip:,.1f}':>15} {'(enter)':>15} {'':>10}")
    print(f"  {'Total GEX':<20} {'$'+f'{total_gex_mil:+,.1f}M':>15} {'(enter)':>15} {'':>10}")
    print(f"  {'Gamma Env.':<20} {'POS' if total_gex > 0 else 'NEG':>15} {'(enter)':>15} {'':>10}")
    print(f"\n  Fill in SpotGamma/SqueezeMetrics values to validate model accuracy.")
    print(f"{'='*75}")


# ------------------------------------------------------------
# Função principal de análise
# ------------------------------------------------------------
async def analyze_options(spot: float, underlying: str = "PETR4"):
       """
       Fetch options data from B3, compute Greeks via Black-Scholes, and analyze.
       Spot is passed as a parameter so the analysis aligns with current price.
       """

       # Fetch options data from B3 and compute Greeks
       df = load_b3_options_data(underlying, spot)
       if df.empty:
           print(f"[X] No options data available for {underlying}")
           return

       # Remove invalid rows
       df = df.dropna(subset=['Strike', 'IV', 'Gamma'])
       df = df[df['Strike'] > 0]             
   
       # Divide o DataFrame entre opções de call e put com base na coluna 'Tipo'.
       # O objetivo é separar as opções para análises específicas por tipo.
       calls = df[df['Tipo'].str.upper().str.contains('CALL')]
       puts  = df[df['Tipo'].str.upper().str.contains('PUT')]
   
       # ------------------------------------------------------------
       # RAZÃO PUT/CALL GLOBAL (medidor de sentimento de mercado)
       # ------------------------------------------------------------
       # Calcula o total de contratos de calls.
       # O objetivo é somar o interesse aberto em calls para comparação com puts.
       total_calls = calls['Tit.'].sum()
       # Calcula o total de contratos de puts.
       total_puts  = puts['Tit.'].sum()
       # Calcula a razão Put/Call global, tratando divisão por zero.
       # O objetivo é obter uma métrica de sentimento: alto PCR indica medo (bearish), baixo indica otimismo (bullish).
       pcr_global = total_puts / total_calls if total_calls > 0 else np.nan
   
       # Imprime o cabeçalho da seção de PCR global.
       print(f"\n===== STOCK OPTIONS — Global PCR =====")
       # Imprime o preço spot atual.
       print(f"Spot: {spot:.2f}")
       # Imprime o total de calls formatado.
       print(f"Total Calls: {total_calls:,.2f}")
       # Imprime o total de puts formatado.
       print(f"Total Puts : {total_puts:,.2f}")
       # Imprime a razão Put/Call formatada.
       print(f"Put/Call Ratio: {pcr_global:.2f}")
   
       # ------------------------------------------------------------
       # SKEW DE VOLATILIDADE IMPLÍCITA — OTM puts vs OTM calls
       # ------------------------------------------------------------
       # Filtra puts out-of-the-money (strikes abaixo do spot).
       # O objetivo é isolar puts OTM para calcular IV média.
       puts_otm  = puts[puts['Strike'] < spot]   
       # Filtra calls out-of-the-money (strikes acima do spot).
       calls_otm = calls[calls['Strike'] > spot] 
       # Calcula a IV média de puts OTM em porcentagem.
       iv_puts_otm  = puts_otm['IV'].mean() * 100
       # Calcula a IV média de calls OTM em porcentagem.
       iv_calls_otm = calls_otm['IV'].mean() * 100
       # Calcula o skew: positivo indica puts mais caros (medo).
       # O objetivo é medir o viés de volatilidade, indicando hedging ou especulação.
       iv_skew = iv_puts_otm - iv_calls_otm      
   
       # Imprime o cabeçalho da seção de skew de IV.
       print(f"\n===== Implied Volatility Skew =====")
       # Imprime IV de puts OTM.
       print(f"OTM Puts IV : {iv_puts_otm:.2f}%")
       # Imprime IV de calls OTM.
       print(f"OTM Calls IV: {iv_calls_otm:.2f}%")
       # Imprime o skew calculado.
       print(f"Skew (Puts - Calls): {iv_skew:.2f}%")
   
       # ------------------------------------------------------------
       # RAZÃO PUT/CALL por faixas de strike (bins ao redor do spot)
       # ------------------------------------------------------------
       # Define as faixas de strike relativas ao spot.
       # O objetivo é categorizar strikes em regiões como deep OTM, near OTM, ATM, etc., para análise segmentada.
       bins = [
           (0, 0.95*spot),          # Deep OTM puts
           (0.95*spot, 0.99*spot),  # Near OTM puts
           (0.99*spot, 1.01*spot),  # ATM range
           (1.01*spot, 1.05*spot),  # Near OTM calls
           (1.05*spot, np.inf),     # Far OTM calls
       ]
       # Inicializa uma lista para armazenar os resultados por faixa.
       rows = []
       # Itera sobre cada faixa de bins.
       for (low, high) in bins:
           # Cria um rótulo para a faixa.
           label = f"{low:.2f}-{high if np.isfinite(high) else '∞'}"
           # Soma contratos de calls na faixa.
           c = calls[(calls['Strike']>=low)&(calls['Strike']<high)]['Tit.'].sum()
           # Soma contratos de puts na faixa.
           p = puts[(puts['Strike']>=low)&(puts['Strike']<high)]['Tit.'].sum()
           # Calcula PCR para a faixa, tratando divisão por zero.
           pcr = p/c if c>0 else np.nan
           # Adiciona a linha à lista.
           rows.append((label, c, p, pcr))
       # Cria um DataFrame com os resultados de PCR por faixa.
       # O objetivo é visualizar o sentimento por regiões de strike.
       df_pcr = pd.DataFrame(rows, columns=['Strike Range','Calls','Puts','PCR'])
       # Imprime o cabeçalho da seção.
       print(f"\n===== PCR by Strike Range =====")
       # Imprime o DataFrame.
       print(df_pcr)
   
       # ------------------------------------------------------------
       # NOTIONAL (volume financeiro por strike)
       # ------------------------------------------------------------
       # Agrupa o volume financeiro por strike e tipo, desempilhando para colunas.
       # O objetivo é calcular o notional por strike para visualização.
       vol_by_strike = df.groupby(['Strike','Tipo'])['VolFin'].sum().unstack(fill_value=0)
       # Plota um gráfico de barras empilhadas para o volume por strike.
       # O objetivo é visualizar o volume financeiro de calls e puts por strike.
       vol_by_strike.plot(kind='bar', stacked=True, figsize=(12,12),
                          color=['#2563EB','#EF4444'], alpha=0.7)
       # Adiciona uma linha vertical no strike mais próximo do spot como âncora visual.
       plt.axvline(np.argmin(np.abs(vol_by_strike.index - spot)), color='black', linestyle='--')
       # Define o título do gráfico.
       plt.title(f"Volume Financeiro por Strike — {underlying}")
       # Define o rótulo do eixo Y.
       plt.ylabel("Volume (R$)")
       # Define o rótulo do eixo X.
       plt.xlabel("Strike")
       # Ajusta o layout para evitar cortes.
       plt.tight_layout()
       # Exibe o gráfico.
       plt.show()
   
       # ------------------------------------------------------------
       # EXPOSIÇÃO GAMMA (Cliente vs Dealer)
       # ------------------------------------------------------------
       # Calcula a exposição gamma do cliente: gamma * (spot^2) * contratos.
       df['GEX_customer'] = df['Gamma'] * (spot**2) * df['Tit.']
       # Ajusta o sinal: positivo para calls, negativo para puts.
       # O objetivo é refletir o impacto direcional da gamma.
       df['GEX_customer'] = df['GEX_customer'] * np.where(df['Tipo'].str.upper().str.contains('CALL'), 1, -1)
       # Calcula a exposição gamma do dealer como oposta à do cliente - isto é uma convenção do mercado
       df['GEX_dealer']   = -df['GEX_customer']   
   
       # Agrega a GEX por strike para cliente e dealer.
       # O objetivo é obter totais por strike para análise e plotagem.
       gex_by_strike = df.groupby('Strike', as_index=False).agg(
           GEX_customer=('GEX_customer','sum'),
           GEX_dealer=('GEX_dealer','sum')
       ).sort_values('Strike')

       # ============================================================
       # GEX FOR NEXT FRIDAY (Weekly Expiration)
       # ============================================================
       # Find the next Friday from today
       from datetime import timedelta
       today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
       days_until_friday = (4 - today.weekday()) % 7  # 4 = Friday
       if days_until_friday == 0:
           days_until_friday = 7  # if today is Friday, use next Friday
       next_friday = today + timedelta(days=days_until_friday)
       next_friday_str = next_friday.strftime('%Y-%m-%d')

       print(f"\n{'='*75}")
       print(f"GEX FOR NEXT FRIDAY ({next_friday_str})")
       print(f"{'='*75}")

       # Filter options expiring on next Friday
       df['Expiration'] = pd.to_datetime(df['Expiration'])
       fri_df = df[df['Expiration'].dt.date == next_friday.date()]

       if fri_df.empty:
           print(f"  No options expiring on {next_friday_str}")
           # Show available expirations for reference
           avail_exp = sorted(df['Expiration'].dt.date.unique())
           print(f"  Available expirations: {[str(d) for d in avail_exp[:10]]}")
       else:
           fri_calls = fri_df[fri_df['Tipo'].str.upper().str.contains('CALL')]
           fri_puts  = fri_df[fri_df['Tipo'].str.upper().str.contains('PUT')]

           fri_gex_by_strike = fri_df.groupby('Strike', as_index=False).agg(
               GEX_customer=('GEX_customer', 'sum'),
               GEX_dealer=('GEX_dealer', 'sum')
           ).sort_values('Strike')

           total_gex = fri_gex_by_strike['GEX_customer'].sum()
           max_gex_strike = fri_gex_by_strike.loc[
               fri_gex_by_strike['GEX_customer'].abs().idxmax(), 'Strike'
           ]

           # Gamma flip (zero crossing nearest to spot)
           fri_gvals = fri_gex_by_strike['GEX_customer'].to_numpy()
           fri_strikes = fri_gex_by_strike['Strike'].to_numpy()
           fri_flip = find_gamma_flip(fri_strikes, fri_gvals, spot)

           # Walls — call wall >= spot (resistance), put wall <= spot (support)
           fri_call_gex = fri_calls.groupby('Strike')['GEX_customer'].sum() if not fri_calls.empty else pd.Series(dtype=float)
           fri_call_above = fri_call_gex[fri_call_gex.index >= spot]
           fri_call_wall = fri_call_above.idxmax() if not fri_call_above.empty else np.nan
           fri_put_gex   = fri_puts.groupby('Strike')['GEX_customer'].sum() if not fri_puts.empty else pd.Series(dtype=float)
           fri_put_below = fri_put_gex[fri_put_gex.index <= spot]
           fri_put_wall  = fri_put_below.abs().idxmax() if not fri_put_below.empty else np.nan

           fri_dte = (next_friday - today).days
           print(f"\n  Next Friday: {next_friday_str} ({fri_dte} DTE)")
           print(f"    Contracts: {len(fri_calls)} calls, {len(fri_puts)} puts")
           print(f"    Total GEX: {total_gex/1e6:>10.2f}M")
           print(f"    Peak GEX strike: {max_gex_strike:.2f}")
           if np.isfinite(fri_flip):
               print(f"    Gamma Flip: {fri_flip:.2f}")
           if np.isfinite(fri_call_wall):
               print(f"    Call Wall:  {fri_call_wall:.2f}")
           if np.isfinite(fri_put_wall):
               print(f"    Put Wall:   {fri_put_wall:.2f}")

           # --- Plot GEX for Next Friday ---
           fig_fri, ax = plt.subplots(figsize=(14, 6))
           ax.set_axisbelow(True)
           fri_s = fri_gex_by_strike['Strike'].to_numpy(dtype=float)
           fri_g = (fri_gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)

           u_fri = np.unique(fri_s)
           if len(u_fri) >= 3:
               bw = np.median(np.diff(u_fri)) * 0.6
           elif len(u_fri) == 2:
               bw = abs(u_fri[1] - u_fri[0]) * 0.6
           else:
               bw = 0.1

           colors = np.where(fri_g >= 0, "#10B981", "#EF4444")
           ax.bar(fri_s, fri_g, width=bw, color=colors,
                  edgecolor="none", alpha=0.6, zorder=3)

           if len(fri_g) > 2:
               sm = pd.Series(fri_g).rolling(3, center=True, min_periods=1).mean().values
               ax.plot(fri_s, sm, color='#3B82F6', lw=2, zorder=4, label='Smoothed GEX')

           ax.axvline(spot, color='green', lw=1.2, zorder=5, label=f'Spot: {spot:.2f}')
           if np.isfinite(fri_flip):
               ax.axvline(fri_flip, color='#F59E0B', lw=1.2, ls='--', zorder=5,
                          label=f"Flip: {fri_flip:.2f}")
           if np.isfinite(fri_call_wall):
               ax.axvline(fri_call_wall, color='#2563EB', ls=':', lw=1.6,
                          label=f"Call Wall: {fri_call_wall:.2f}")
               ax.annotate(f"Call Wall\n{fri_call_wall:.2f}",
                           xy=(fri_call_wall, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1),
                           xytext=(8, -18), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='#2563EB',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2563EB', alpha=0.85),
                           ha='left', va='top')
           if np.isfinite(fri_put_wall):
               ax.axvline(fri_put_wall, color='#DC2626', ls='--', lw=1.6,
                          label=f"Put Wall: {fri_put_wall:.2f}")
               ax.annotate(f"Put Wall\n{fri_put_wall:.2f}",
                           xy=(fri_put_wall, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1),
                           xytext=(-8, 18), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='#DC2626',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DC2626', alpha=0.85),
                           ha='right', va='bottom')

           cw_str = f"{fri_call_wall:.2f}" if np.isfinite(fri_call_wall) else "N/A"
           pw_str = f"{fri_put_wall:.2f}" if np.isfinite(fri_put_wall) else "N/A"
           ax.set_title(f"{underlying} — GEX Next Friday ({next_friday_str}, {fri_dte} DTE)"
                        f"  |  Call Wall: {cw_str}  |  Put Wall: {pw_str}",
                        fontsize=12, fontweight='bold')
           ax.set_xlabel('Strike Price')
           ax.set_ylabel('GEX (millions)')
           ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
           ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
           ax.grid(alpha=0.25)
           plt.tight_layout()
           plt.show()
   
       # ------------------------------------------------------------
       # PAREDES DE CALL/PUT — strike com máxima exposição gamma por lado
       # ------------------------------------------------------------
       # Filtra calls e puts incluindo a coluna GEX_customer recém-calculada.
       gex_calls = df[df['Tipo'].str.upper().str.contains('CALL')]
       gex_puts  = df[df['Tipo'].str.upper().str.contains('PUT')]
       # Call wall = strike >= spot com maior GEX de calls (resistência via dealer hedging).
       call_gex_by_strike = gex_calls.groupby('Strike')['GEX_customer'].sum()
       call_gex_above = call_gex_by_strike[call_gex_by_strike.index >= spot]
       call_wall = call_gex_above.idxmax() if not call_gex_above.empty else np.nan
       # Put wall = strike <= spot com maior |GEX| de puts (suporte via dealer hedging).
       put_gex_by_strike = gex_puts.groupby('Strike')['GEX_customer'].sum()
       put_gex_below = put_gex_by_strike[put_gex_by_strike.index <= spot]
       put_wall  = put_gex_below.abs().idxmax() if not put_gex_below.empty else np.nan
   
       # ------------------------------------------------------------
       # GAMMA FLIP — cruzamento de zero na GEX do cliente mais próximo do spot
       # ------------------------------------------------------------
       # Identifica o ponto onde a GEX muda de sinal (positivo->negativo ou vice-versa)
       # usando interpolação linear entre os pontos de mudança de sinal.
       gvals = gex_by_strike['GEX_customer'].to_numpy()
       strikes = gex_by_strike['Strike'].to_numpy()
       gamma_flip = find_gamma_flip(strikes, gvals, spot)
   
       # Imprime o cabeçalho da seção de paredes.
       print(f"\n===== Call/Put Walls =====")
       # Imprime a parede de call.
       print(f"Call Wall: {call_wall:.2f}")
       # Imprime a parede de put.
       print(f"Put  Wall: {put_wall:.2f}")
       # Imprime o gamma flip aproximado.
       print(f"Gamma Flip (approx): {gamma_flip:.2f}")
   
       # ------------------------------------------------------------
       # GRÁFICO DE EXPOSIÇÃO GAMMA (mapa visual de posicionamento)
       # ------------------------------------------------------------
       # Extrai strikes e valores de GEX em milhões.
       strikes = gex_by_strike['Strike'].to_numpy(dtype=float)
       gvals = (gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)
   
       # Calcula largura dinâmica das barras proporcional ao espaçamento de strikes.
       # O objetivo é evitar sobreposição em gráficos com strikes irregulares.
       u = np.unique(strikes)
       if len(u) >= 3:
           step = np.median(np.diff(u))
       elif len(u) == 2:
           step = abs(u[1] - u[0])
       else:
           step = 0.1
       bar_width = step * 0.6  
   
       # Aplica suavização de 3 pontos na GEX.
       smooth = pd.Series(gvals).rolling(3, center=True, min_periods=1).mean().values
   
       # Cria uma figura e eixo para o gráfico.
       fig, ax = plt.subplots(figsize=(10, 10))
       # Coloca o grid abaixo das barras.
       ax.set_axisbelow(True)
   
       # Define cores das barras: verde para gamma positiva, vermelho para negativa.
       bar_colors = np.where(gvals >= 0, "#10B981", "#EF4444")
       # Plota as barras de GEX por strike.
       # O objetivo é visualizar a distribuição de gamma.
       ax.bar(strikes, gvals, width=bar_width, align="center",
              color=bar_colors, edgecolor="none", alpha=0.55, zorder=3,
              label="Gamma Exposure by Strike")
   
       # Plota a linha suavizada para interpretação visual mais fácil.
       ax.plot(strikes, smooth, color="#2563EB", lw=2.2, zorder=4,
               label="Aggregate Gamma Exposure (smoothed)")
   
       # Adiciona marcadores verticais: spot, gamma flip, paredes.
       # O objetivo é destacar níveis chave no gráfico.
       ax.axvline(spot, color="green", lw=1.2, zorder=5, label="Spot")
       if np.isfinite(gamma_flip):
           ax.axvline(gamma_flip, color="#DC2626", lw=1.2, zorder=5,
                      label=f"Gamma Flip (approx): {gamma_flip:.2f}")
   
       # Adiciona sombreamento para regimes de gamma positiva vs negativa.
       if len(strikes):
           x_min, x_max = strikes.min(), strikes.max()
           if np.isfinite(gamma_flip):
               # Sombreia a região de gamma positiva (dealers dampen).
               ax.axvspan(x_min, gamma_flip, color="#E5F3FF", alpha=0.35,
                          label="Positive Gamma: dealers dampen moves")
               # Sombreia a região de gamma negativa (dealers amplify).
               ax.axvspan(gamma_flip, x_max, color="#FEE2E2", alpha=0.35,
                          label="Negative Gamma: dealers amplify moves")
   
       # Ajusta a escala do eixo Y de forma adaptativa.
       # O objetivo é garantir que o gráfico seja bem dimensionado.
       ymin = float(np.nanmin(gvals)) if len(gvals) else -1.0
       ymax = float(np.nanmax(gvals)) if len(gvals) else  1.0
       if ymin < 0 and ymax > 0:
           lim = max(abs(ymin), abs(ymax))*1.25
           ax.set_ylim(-lim, lim)
       else:
           pad = 0.15*(ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
           ax.set_ylim(ymin - pad, ymax + pad)
   
       # Formata o eixo Y com separadores de milhar.
       ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
       # Define rótulo do eixo X.
       ax.set_xlabel("Strike Price")
       # Define rótulo do eixo Y.
       ax.set_ylabel("Gamma Exposure (USD, millions)")
       # Define título do gráfico.
       ax.set_title(f"Gamma Exposure by Strike — {underlying}")  # Nota: título menciona PETR4, mas código é para BOVA11; possivelmente um erro.
   
       # Adiciona linhas para paredes de call e put.
       if np.isfinite(call_wall):
           ax.axvline(call_wall, color="#374151", linestyle=":",  lw=1.6,
                       label=f"Call Wall: {call_wall:.2f}")
       if np.isfinite(put_wall):
           ax.axvline(put_wall,  color="#9CA3AF", linestyle="--", lw=1.6,
                       label=f"Put Wall: {put_wall:.2f}")
   
       # Adiciona legenda, grid e marca d'água.
       # O objetivo é tornar o gráfico informativo e profissional.
       ax.legend(loc="upper right", ncol=1, fontsize=9, framealpha=0.95)
       fig.text(0.5, 0.96, "om-qs.com", ha="center", va="center", fontsize=9, alpha=0.7)
       ax.grid(alpha=0.25)
       # Ajusta o layout.
       plt.tight_layout(rect=[0, 0, 1, 0.94])
       # Exibe o gráfico.
       plt.show()
   
       # ------------------------------------------------------------
       # MÉTRICAS ESTENDIDAS DE ESTRUTURA — Interpretação qualitativa
       # ------------------------------------------------------------
       # Imprime cabeçalho da seção estendida.
       print("\n" + "="*75)
       print("EXTENDED MARKET STRUCTURE METRICS — STOCK TRACE-Lite View")
       print("="*75)
   
       # --- Recomputa gamma flip suavizado para consistência
       # Extrai strikes e GEX.
       strikes = gex_by_strike["Strike"].to_numpy(dtype=float)
       gvals   = gex_by_strike["GEX_customer"].to_numpy(dtype=float)
       # Aplica média móvel de 5 pontos.
       smooth  = pd.Series(gvals).rolling(5, center=True, min_periods=1).mean().values
       # Encontra gamma flip na curva mais suavizada.
       gamma_flip = strikes[np.argmin(np.abs(smooth))] if len(strikes) else np.nan
   
       # --- Recap de PCR global
       # Armazena OI de calls e puts.
       calls_oi = total_calls
       puts_oi  = total_puts
       pcr_oi   = pcr_global
   
       # Imprime PCR baseado em OI.
       print(f"Put/Call Ratio (OI):  {pcr_oi:>6.2f}")
       # Classifica o sentimento com base no PCR.
       # O objetivo é fornecer uma interpretação qualitativa do sentimento de mercado.
       if 0.9 <= pcr_oi <= 1.1:
           sentiment = "Neutral"
       elif pcr_oi > 1.1:
           sentiment = "Bearish — put demand dominates"
       else:
           sentiment = "Bullish — call demand dominates"
       # Imprime o sentimento.
       print(f"Sentiment:            {sentiment}")
   
       # --- Interpretação de skew de volatilidade
       print("\nVolatility Skew:")
       # Imprime IV de puts OTM.
       print(f"IV (OTM Puts):   {iv_puts_otm:>6.2f}%")
       # Imprime IV de calls OTM.
       print(f"IV (OTM Calls):  {iv_calls_otm:>6.2f}%")
       # Imprime skew.
       print(f"Skew (Puts−Calls): {iv_skew:>6.2f}%")
   
       # Interpreta o skew qualitativamente.
       if iv_skew > 10:
           print("Interpretation:  Elevated skew — investors hedging downside risk.")
       elif iv_skew < 0:
           print("Interpretation:  Inverted skew — speculative upside bias.")
       else:
           print("Interpretation:  Balanced implied vol surface.")
   
       # --- Análise de gamma flip
       print("\nGamma Flip Analysis:")
       # Imprime gamma flip.
       print(f"Gamma Flip (approx): {gamma_flip:>8.2f}")
       # Imprime spot.
       print(f"Spot:                 {spot:>8.2f}")
   
       # Calcula diferença relativa ao flip.
       if np.isfinite(gamma_flip):
           diff = spot - gamma_flip
           pct  = diff / gamma_flip * 100
           side = "above" if diff > 0 else "below"
           # Imprime posição relativa.
           print(f"Spot is {abs(pct):.2f}% {side} the flip.")
           # Interpreta o impacto dos dealers.
           if diff > 0:
               print("→ Dealers short gamma: market mechanically amplified.")
           else:
               print("→ Dealers long gamma: market mechanically dampened.")
   
       # --- Classificação de regime de mercado
       # Classifica o regime com base na posição relativa ao gamma flip.
       # O objetivo é sugerir estratégias baseadas no regime detectado.
       if np.isfinite(gamma_flip):
           if spot >= gamma_flip * 1.05:
               regime = "HIGH VOLATILITY"
               rationale = "Dealers short gamma, hedging exacerbates moves."
               strategy = "Long gamma, directional or convexity-driven setups."
           elif spot <= gamma_flip * 0.95:
               regime = "LOW VOLATILITY"
               rationale = "Dealers long gamma, hedging absorbs shocks."
               strategy = "Range trading, vol selling, short gamma spreads."
           else:
               regime = "TRANSITION ZONE"
               rationale = "Market near flip — unstable hedging behavior."
               strategy = "Neutral, calendar, or butterfly setups."
       else:
           regime, rationale, strategy = "UNKNOWN", "Gamma Flip not found", "N/A"
   
       # Imprime o regime detectado.
       print("\nMarket Regime:")
       print(f"Detected:     {regime}")
       # Imprime a rationale.
       print(f"Rationale:    {rationale}")
       # Imprime a estratégia recomendada.
       print(f"Recommended:  {strategy}")
   
       # --- Zonas significativas de GEX inferidas
       print("\nSignificant GEX Zones:")
       # Ordena GEX por valor descendente.
       gex_sorted = gex_by_strike.sort_values("GEX_customer", ascending=False)
       # Seleciona top 4 para resistance (gamma positiva).
       resist = gex_sorted.head(4)   
       # Seleciona bottom 4 para support (gamma negativa).
       supports = gex_sorted.tail(4) 
   
       # Imprime zonas de suporte.
       # O objetivo é identificar níveis onde dealers podem amortecer movimentos.
       print("Support Zones (dealers long gamma → cushion):")
       for _, r in supports.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           # Classifica força com base no valor absoluto.
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | {gex_mil:>7.2f}M | {strength}")
   
       # Imprime zonas de resistance.
       # O objetivo é identificar níveis onde movimentos podem ser amplificados.
       print("\nResistance Zones (dealers short gamma → acceleration risk):")
       for _, r in resist.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | +{gex_mil:>7.2f}M | {strength}")
   
       # --- Snapshot de resumo
       # Imprime um resumo final com níveis chave.
       # O objetivo é fornecer uma visão rápida das métricas principais.
       print("\nSummary Snapshot:")
       print(f"Spot:        {spot:,.2f}")
       print(f"Call Wall:   {call_wall:,.2f}")
       print(f"Put Wall:    {put_wall:,.2f}")
       print(f"Gamma Flip:  {gamma_flip:,.2f}")
       print(f"Market Regime: {regime}")
       print("="*75)
       # Imprime as primeiras linhas do DataFrame para verificação.
       print(df.head())
       print("="*75)
       # Imprime as colunas do DataFrame.
       print(df.columns)

async def main():
    if VALIDATION_MODE == "SPY":
        # SPY Validation Mode — no MT5 needed
        print(f"Running SPY GEX Validation Mode...")
        await analyze_spy_validation()
    else:
        # BOVA11 Mode — requires MetaTrader 5
        mt5_conn = MT5Connector()
        for asset in ASSET_SYMBOL:
            print(f"\n{'#'*80}\nAnalyzing {asset}...\n{'#'*80}")
            symbol_info = mt5_conn.get_symbol_info(asset)
            spot_price = (symbol_info.bid + symbol_info.ask) / 2
            print(f"Analyzing options data for {asset} with spot price {spot_price:.2f}...")
            await analyze_options(spot_price, asset)
# ------------------------------------------------------------
# Exemplo de uso (descomente para executar)
# ------------------------------------------------------------
asyncio.run(main())