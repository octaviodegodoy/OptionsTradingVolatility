# -*- coding: utf-8 -*-
"""
B3 Options Data Loader
----------------------
Fetches B3 COTAHIST data, classifies call/put, computes Greeks via
Black-Scholes, and merges real OI when available.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from bs_greeks import (
    RISK_FREE_RATE,
    bs_gamma,
    bs_delta,
    implied_vol,
)

# Resolve paths so get_b3_data is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from get_b3_data import fetch_b3_historical_file, fetch_open_interest


def load_b3_options_data(underlying, spot, date=None):
    """
    Fetch options from B3 historical file and compute Greeks via Black-Scholes.
    Returns DataFrame with columns:
        Ticker, Tipo, Strike, Ultimo, IV, Delta, Gamma, Tit., VolFin
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
        'Tit.': options['quantity'].values.astype(float),
        'VolFin': options['volume'].values.astype(float),
        'DTE': options['DTE'].values.astype(int),
        'Expiration': options['Expiration'].values,
    })

    # ---- Merge real OI data if available ----
    oi_source = 'daily_volume'
    try:
        oi_data = fetch_open_interest(
            underlying=underlying,
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
