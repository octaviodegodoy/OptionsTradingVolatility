# -*- coding: utf-8 -*-
"""
Options GEX Analytics — Outspoken Market version
-------------------------------------------------
BOVA11 — B3 Brazilian options via COTAHIST + OI proxy

Performs:
- Global and range-based Put/Call Ratio
- IV skew (OTM puts vs OTM calls)
- Notional by strike (volume financeiro)
- Gamma Exposure (Customer/Dealer)
- Call/Put walls and Gamma Flip
- $IND ↔ BOVA11 Kalman regression & delta-neutral hedge sizing

Practical Usage — Intraday $IND Trading
----------------------------------------
Best days to run:
  Mon/Tue → most reliable GEX levels (full gamma profile after Friday expiry).
  Wed     → mid-week check; levels still hold well.
  Thu/Fri → levels degrade as short-dated gamma dominates; use Friday GEX section.

Recommended intraday timeframe: 15-minute bars.
  - Dealer hedging rebalances are visible at this granularity.
  - Clean wall tests (call wall = resistance, put wall = support).
  - Drop to 5-min if spot is within ±0.5% of gamma flip (transition zone).

Session workflow:
  1. Pre-market (09:00 BRT): run script → note $IND call wall, put wall, gamma flip.
  2. 10:00–11:30: first 6 bars — price discovery vs GEX levels.
  3. Wall touch on 15-min close → mean-reversion entry (positive gamma regime).
  4. Wall break on 15-min close → trend continuation (negative gamma regime).
  5. 14:00–16:00: strongest dealer hedging flow; 15-min signals most reliable.
"""
import numpy as np
import pandas as pd
import os
import sys
import asyncio
from datetime import datetime, timedelta

# Ensure parent dir is on sys.path for mt5_connector / get_b3_data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from constants import ASSET_SYMBOL
from gex_utils import find_gamma_flip
from gex_plots import plot_notional_by_strike, plot_gex_friday, plot_gex_all_expiry
from b3_options_loader import load_b3_options_data

from kalman_price_mapper import (
    KalmanPriceMapper,
    build_ind_bova11_mapper,
    calculate_delta_neutral_hedge,
)

from mt5_connector import MT5Connector


async def analyze_options(spot: float, underlying: str = "PETR4", ind_mapper: KalmanPriceMapper = None, show_plots: bool = False):
       """
       Fetch options data from B3, compute Greeks via Black-Scholes, and analyze.
       Spot is passed as a parameter so the analysis aligns with current price.
       If ind_mapper is provided, $IND equivalents are printed alongside BOVA11 levels.
       If show_plots is False, all matplotlib charts are suppressed.
       """

       df = load_b3_options_data(underlying, spot)
       if df.empty:
           print(f"[X] No options data available for {underlying}")
           return

       df = df.dropna(subset=['Strike', 'IV', 'Gamma'])
       df = df[df['Strike'] > 0]

       calls = df[df['Tipo'].str.upper().str.contains('CALL')]
       puts  = df[df['Tipo'].str.upper().str.contains('PUT')]

       # ------------------------------------------------------------
       # PUT/CALL RATIO (global sentiment)
       # ------------------------------------------------------------
       total_calls = calls['Tit.'].sum()
       total_puts  = puts['Tit.'].sum()
       pcr_global = total_puts / total_calls if total_calls > 0 else np.nan

       print(f"\n===== STOCK OPTIONS — Global PCR =====")
       print(f"Spot: {spot:.2f}")
       print(f"Total Calls: {total_calls:,.2f}")
       print(f"Total Puts : {total_puts:,.2f}")
       print(f"Put/Call Ratio: {pcr_global:.2f}")

       # ------------------------------------------------------------
       # IV SKEW — OTM puts vs OTM calls
       # ------------------------------------------------------------
       puts_otm  = puts[puts['Strike'] < spot]
       calls_otm = calls[calls['Strike'] > spot]
       iv_puts_otm  = puts_otm['IV'].mean() * 100
       iv_calls_otm = calls_otm['IV'].mean() * 100
       iv_skew = iv_puts_otm - iv_calls_otm

       print(f"\n===== Implied Volatility Skew =====")
       print(f"OTM Puts IV : {iv_puts_otm:.2f}%")
       print(f"OTM Calls IV: {iv_calls_otm:.2f}%")
       print(f"Skew (Puts - Calls): {iv_skew:.2f}%")

       # ------------------------------------------------------------
       # PCR BY STRIKE RANGE
       # ------------------------------------------------------------
       bins = [
           (0, 0.95*spot),          # Deep OTM puts
           (0.95*spot, 0.99*spot),  # Near OTM puts
           (0.99*spot, 1.01*spot),  # ATM range
           (1.01*spot, 1.05*spot),  # Near OTM calls
           (1.05*spot, np.inf),     # Far OTM calls
       ]
       rows = []
       for (low, high) in bins:
           label = f"{low:.2f}-{high if np.isfinite(high) else '∞'}"
           c = calls[(calls['Strike']>=low)&(calls['Strike']<high)]['Tit.'].sum()
           p = puts[(puts['Strike']>=low)&(puts['Strike']<high)]['Tit.'].sum()
           pcr = p/c if c>0 else np.nan
           rows.append((label, c, p, pcr))
       df_pcr = pd.DataFrame(rows, columns=['Strike Range','Calls','Puts','PCR'])
       print(f"\n===== PCR by Strike Range =====")
       print(df_pcr)

       # ------------------------------------------------------------
       # NOTIONAL (volume financeiro por strike)
       # ------------------------------------------------------------
       vol_by_strike = df.groupby(['Strike','Tipo'])['VolFin'].sum().unstack(fill_value=0)
       plot_notional_by_strike(vol_by_strike, spot, underlying, show_plots)

       # ------------------------------------------------------------
       # GAMMA EXPOSURE (Customer)
       # ------------------------------------------------------------
       df['GEX_customer'] = df['Gamma'] * (spot**2) * df['Tit.']
       df['GEX_customer'] = df['GEX_customer'] * np.where(df['Tipo'].str.upper().str.contains('CALL'), 1, -1)
   
       gex_by_strike = df.groupby('Strike', as_index=False).agg(
           GEX_customer=('GEX_customer','sum')
       ).sort_values('Strike')

       # ============================================================
       # GEX FOR NEXT FRIDAY (Weekly Expiration)
       # ============================================================
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
               GEX_customer=('GEX_customer', 'sum')
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

           plot_gex_friday(fri_gex_by_strike, spot, underlying,
                          next_friday_str, fri_dte, fri_flip,
                          fri_call_wall, fri_put_wall, show_plots)

       # Call/Put Walls — strike with max gamma exposure per side
       gex_calls = df[df['Tipo'].str.upper().str.contains('CALL')]
       gex_puts  = df[df['Tipo'].str.upper().str.contains('PUT')]

       call_gex_by_strike = gex_calls.groupby('Strike')['GEX_customer'].sum()
       call_gex_above = call_gex_by_strike[call_gex_by_strike.index >= spot]
       call_wall = call_gex_above.idxmax() if not call_gex_above.empty else np.nan

       put_gex_by_strike = gex_puts.groupby('Strike')['GEX_customer'].sum()
       put_gex_below = put_gex_by_strike[put_gex_by_strike.index <= spot]
       put_wall  = put_gex_below.abs().idxmax() if not put_gex_below.empty else np.nan

       # Gamma Flip — zero crossing of customer GEX nearest to spot
       gvals = gex_by_strike['GEX_customer'].to_numpy()
       strikes = gex_by_strike['Strike'].to_numpy()
       gamma_flip = find_gamma_flip(strikes, gvals, spot)

       print(f"\n===== Call/Put Walls =====")
       print(f"Call Wall: {call_wall:.2f}")
       print(f"Put  Wall: {put_wall:.2f}")
       print(f"Gamma Flip (approx): {gamma_flip:.2f}")

       plot_gex_all_expiry(gex_by_strike, spot, underlying, gamma_flip,
                          call_wall, put_wall, show_plots)
   
       # Extended Market Structure Metrics
       print("\n" + "="*75)
       print("EXTENDED MARKET STRUCTURE METRICS — STOCK TRACE-Lite View")
       print("="*75)
   
       print(f"Put/Call Ratio (OI):  {pcr_global:>6.2f}")
       if 0.9 <= pcr_global <= 1.1:
           sentiment = "Neutral"
       elif pcr_global > 1.1:
           sentiment = "Bearish — put demand dominates"
       else:
           sentiment = "Bullish — call demand dominates"
       print(f"Sentiment:            {sentiment}")
   
       print("\nVolatility Skew:")
       print(f"IV (OTM Puts):   {iv_puts_otm:>6.2f}%")
       print(f"IV (OTM Calls):  {iv_calls_otm:>6.2f}%")
       print(f"Skew (Puts−Calls): {iv_skew:>6.2f}%")
   
       if iv_skew > 10:
           print("Interpretation:  Elevated skew — investors hedging downside risk.")
       elif iv_skew < 0:
           print("Interpretation:  Inverted skew — speculative upside bias.")
       else:
           print("Interpretation:  Balanced implied vol surface.")
   
       print("\nGamma Flip Analysis:")
       print(f"Gamma Flip (approx): {gamma_flip:>8.2f}")
       print(f"Spot:                 {spot:>8.2f}")
   
       if np.isfinite(gamma_flip):
           diff = spot - gamma_flip
           pct  = diff / gamma_flip * 100
           side = "above" if diff > 0 else "below"
           print(f"Spot is {abs(pct):.2f}% {side} the flip.")
           if diff > 0:
               print("→ Dealers short gamma: market mechanically amplified.")
           else:
               print("→ Dealers long gamma: market mechanically dampened.")
   
       # Market regime classification
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
   
       print("\nMarket Regime:")
       print(f"Detected:     {regime}")
       print(f"Rationale:    {rationale}")
       print(f"Recommended:  {strategy}")
   
       # Significant GEX zones
       print("\nSignificant GEX Zones:")
       gex_sorted = gex_by_strike.sort_values("GEX_customer", ascending=False)
       resist = gex_sorted.head(4)
       supports = gex_sorted.tail(4)
   
       print("Support Zones (dealers long gamma → cushion):")
       for _, r in supports.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | {gex_mil:>7.2f}M | {strength}")
   
       print("\nResistance Zones (dealers short gamma → acceleration risk):")
       for _, r in resist.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | +{gex_mil:>7.2f}M | {strength}")
   
       # Summary Snapshot
       print("\nSummary Snapshot:")
       if ind_mapper is not None:
           ind_spot       = ind_mapper.bova11_to_ind(spot)
           ind_call_wall  = ind_mapper.bova11_to_ind(call_wall)
           ind_put_wall   = ind_mapper.bova11_to_ind(put_wall)
           ind_gamma_flip = ind_mapper.bova11_to_ind(gamma_flip)
           print(f"{'':>15} {'BOVA11':>12} {'$IND':>14}")
           print(f"  {'Spot':<13} {spot:>12,.2f} {ind_spot:>14,.0f}")
           print(f"  {'Call Wall':<13} {call_wall:>12,.2f} {ind_call_wall:>14,.0f}")
           print(f"  {'Put Wall':<13} {put_wall:>12,.2f} {ind_put_wall:>14,.0f}")
           print(f"  {'Gamma Flip':<13} {gamma_flip:>12,.2f} {ind_gamma_flip:>14,.0f}")
           print(f"  Kalman β = {ind_mapper.beta:,.4f}  α = {ind_mapper.alpha:,.2f}")

           # Delta-neutral hedge estimate
           print(f"\n  Delta-Neutral Hedge (1 WIN long → buy BOVA11 puts):")
           try:
               dn = calculate_delta_neutral_hedge(ind_mapper, df, spot,
                                                  win_contracts=1, side='put')
               print(f"    Option:     {dn['ticker']}  (strike {dn['strike']:.2f}, "
                     f"DTE {dn['dte']}, IV {dn['iv']:.1%})")
               print(f"    Δ option:   {dn['option_delta']:+.4f}")
               print(f"    Qty needed: {dn['n_options']} puts")
               print(f"    Net Δ:      {dn['net_delta']:+.4f}")
               print(f"    $IND strike: {dn['ind_strike']:,.0f}")
           except Exception as e:
               print(f"    [!] Could not estimate: {e}")

       else:
           print(f"Spot:        {spot:,.2f}")
           print(f"Call Wall:   {call_wall:,.2f}")
           print(f"Put Wall:    {put_wall:,.2f}")
           print(f"Gamma Flip:  {gamma_flip:,.2f}")
       print(f"Market Regime: {regime}")
       print("="*75)

async def main():
    mt5_conn = MT5Connector()

    # Build Kalman mapper: BOVA11 ↔ $IND
    try:
        ind_mapper = build_ind_bova11_mapper(mt5_conn)
    except Exception as e:
        print(f"[!] Could not build IND↔BOVA11 mapper: {e}")
        ind_mapper = None

    for asset in ASSET_SYMBOL:
        print(f"\n{'#'*80}\nAnalyzing {asset}...\n{'#'*80}")
        symbol_info = mt5_conn.get_symbol_info(asset)
        spot_price = (symbol_info.bid + symbol_info.ask) / 2
        print(f"Analyzing options data for {asset} with spot price {spot_price:.2f}...")
        # Pass ind_mapper only when analyzing BOVA11
        mapper_for_asset = ind_mapper if asset == "BOVA11" else None
        await analyze_options(spot_price, asset, ind_mapper=mapper_for_asset, show_plots=True)

asyncio.run(main())