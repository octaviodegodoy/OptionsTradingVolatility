"""
GEX Complete Pipeline:
  1. Fetch/build options chain
  2. Calculate GEX
  3. Aggregate key levels
  4. Write CSV to MT5/Files/
  5. Install MQL5 indicator

Everything connected. Just run: python gex_to_mt5_complete.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️  MetaTrader5 not installed. Install with: pip install MetaTrader5")


# ============================================================
# 1. Black-Scholes Gamma
# ============================================================
def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# ============================================================
# 2. GEX Calculation per Strike
# ============================================================
def calculate_gex_by_strike(options_chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    now = datetime.now()
    results = []

    for _, row in options_chain.iterrows():
        T = (row['expiration_date'] - now).days / 365.0
        if T <= 0:
            continue

        r = 0.05
        gamma_call = bs_gamma(spot, row['strike'], T, r, row['call_iv'])
        gamma_put = bs_gamma(spot, row['strike'], T, r, row['put_iv'])

        contract_multiplier = 100
        gex_call = row['call_oi'] * gamma_call * contract_multiplier * spot
        gex_put = -row['put_oi'] * gamma_put * contract_multiplier * spot

        results.append({
            'strike': row['strike'],
            'gamma_call': gamma_call,
            'gamma_put': gamma_put,
            'call_oi': row['call_oi'],
            'put_oi': row['put_oi'],
            'gex_call': gex_call,
            'gex_put': gex_put,
            'gex_total': gex_call + gex_put,
        })

    return pd.DataFrame(results)


# ============================================================
# 3. Aggregate GEX — Key Levels
# ============================================================
def aggregate_gex(gex_df: pd.DataFrame) -> dict:
    by_strike = gex_df.groupby('strike').agg({
        'gex_call': 'sum',
        'gex_put': 'sum',
        'gex_total': 'sum'
    }).reset_index()

    total_gex = by_strike['gex_total'].sum()

    # Gamma Flip
    by_strike_sorted = by_strike.sort_values('strike')
    cumulative = by_strike_sorted['gex_total'].cumsum()
    sign_changes = np.where(np.diff(np.sign(cumulative)))[0]

    gamma_flip = None
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        gamma_flip = by_strike_sorted['strike'].iloc[idx]

    # Call Wall / Put Wall
    call_wall = by_strike.loc[by_strike['gex_call'].idxmax(), 'strike']
    put_wall = by_strike.loc[by_strike['gex_put'].idxmin(), 'strike']

    return {
        'total_gex': total_gex,
        'gamma_flip': gamma_flip,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gex_by_strike': by_strike,
        'regime': 'POSITIVE' if total_gex > 0 else 'NEGATIVE',
    }


# ============================================================
# 4. Write GEX to MT5 Files Directory
# ============================================================
def send_gex_to_mt5_via_file(gex_levels: dict, symbol: str = "PETR4") -> bool:
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 package not available")
        return False

    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False

    terminal_info = mt5.terminal_info()
    data_path = terminal_info.data_path
    mt5.shutdown()

    # MQL5 can ONLY read from MQL5/Files/
    files_path = os.path.join(data_path, "MQL5", "Files")
    os.makedirs(files_path, exist_ok=True)

    # ── Write CSV ──
    csv_path = os.path.join(files_path, f"GEX_{symbol}.csv")
    with open(csv_path, 'w') as f:
        f.write("key,value\n")
        f.write(f"total_gex,{gex_levels['total_gex']}\n")
        f.write(f"gamma_flip,{gex_levels['gamma_flip'] or 0}\n")
        f.write(f"call_wall,{gex_levels['call_wall']}\n")
        f.write(f"put_wall,{gex_levels['put_wall']}\n")
        f.write(f"regime,{1.0 if gex_levels['total_gex'] > 0 else -1.0}\n")

    print(f"✅ CSV written: {csv_path}")

    # ── Write JSON (backup/alternative) ──
    json_path = os.path.join(files_path, f"GEX_{symbol}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'total_gex': float(gex_levels['total_gex']),
            'gamma_flip': float(gex_levels['gamma_flip'] or 0),
            'call_wall': float(gex_levels['call_wall']),
            'put_wall': float(gex_levels['put_wall']),
            'regime': 1.0 if gex_levels['total_gex'] > 0 else -1.0,
            'updated_at': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"✅ JSON written: {json_path}")
    return True


# ============================================================
# 5. Install MQL5 Indicator into MT5
# ============================================================
def install_mql5_indicator(symbol: str = "PETR4") -> bool:
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 package not available")
        return False

    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False

    terminal_info = mt5.terminal_info()
    data_path = terminal_info.data_path
    mt5.shutdown()

    indicators_path = os.path.join(data_path, "MQL5", "Indicators", "GEX")
    os.makedirs(indicators_path, exist_ok=True)

    mql5_code = f'''//+------------------------------------------------------------------+
//| GEX_Levels.mq5 — Reads GEX data from CSV file written by Python |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_plots 0

input string Symbol_Name = "{symbol}";
input int    Update_Seconds = 5;

double gamma_flip, call_wall, put_wall, gex_regime, total_gex;

int OnInit()
{{
    EventSetTimer(Update_Seconds);
    ReadGEXFile();
    return(INIT_SUCCEEDED);
}}

void ReadGEXFile()
{{
    string filename = "GEX_" + Symbol_Name + ".csv";
    int handle = FileOpen(filename, FILE_READ | FILE_CSV | FILE_ANSI, ',');

    if(handle == INVALID_HANDLE)
    {{
        Print("Cannot open: ", filename, " Error: ", GetLastError());
        Comment("GEX: Waiting for data...\\nRun Python to generate CSV.");
        return;
    }}

    // Skip header
    FileReadString(handle); // "key"
    FileReadString(handle); // "value"

    while(!FileIsEnding(handle))
    {{
        string key = FileReadString(handle);
        double value = StringToDouble(FileReadString(handle));

        if(key == "total_gex")   total_gex   = value;
        if(key == "gamma_flip")  gamma_flip  = value;
        if(key == "call_wall")   call_wall   = value;
        if(key == "put_wall")    put_wall    = value;
        if(key == "regime")      gex_regime  = value;
    }}

    FileClose(handle);
    DrawLevels();
}}

void DrawLevels()
{{
    ObjectDelete(0, "GEX_GammaFlip");
    ObjectDelete(0, "GEX_CallWall");
    ObjectDelete(0, "GEX_PutWall");

    if(gamma_flip > 0)
    {{
        ObjectCreate(0, "GEX_GammaFlip", OBJ_HLINE, 0, 0, gamma_flip);
        ObjectSetInteger(0, "GEX_GammaFlip", OBJPROP_COLOR, clrYellow);
        ObjectSetInteger(0, "GEX_GammaFlip", OBJPROP_WIDTH, 2);
        ObjectSetInteger(0, "GEX_GammaFlip", OBJPROP_STYLE, STYLE_DASH);
        ObjectSetString(0, "GEX_GammaFlip", OBJPROP_TEXT,
                        "Gamma Flip: " + DoubleToString(gamma_flip, 2));
    }}

    ObjectCreate(0, "GEX_CallWall", OBJ_HLINE, 0, 0, call_wall);
    ObjectSetInteger(0, "GEX_CallWall", OBJPROP_COLOR, clrRed);
    ObjectSetInteger(0, "GEX_CallWall", OBJPROP_WIDTH, 2);
    ObjectSetString(0, "GEX_CallWall", OBJPROP_TEXT,
                    "Call Wall (Res): " + DoubleToString(call_wall, 2));

    ObjectCreate(0, "GEX_PutWall", OBJ_HLINE, 0, 0, put_wall);
    ObjectSetInteger(0, "GEX_PutWall", OBJPROP_COLOR, clrLime);
    ObjectSetInteger(0, "GEX_PutWall", OBJPROP_WIDTH, 2);
    ObjectSetString(0, "GEX_PutWall", OBJPROP_TEXT,
                    "Put Wall (Sup): " + DoubleToString(put_wall, 2));

    string regime_str = (gex_regime > 0) ? "POSITIVE (Low Vol)" : "NEGATIVE (High Vol)";
    Comment("=== GEX DASHBOARD ===\\n"
          + "Regime: " + regime_str + "\\n"
          + "Total GEX: " + DoubleToString(total_gex, 0) + "\\n"
          + "Gamma Flip: " + DoubleToString(gamma_flip, 2) + "\\n"
          + "Call Wall: " + DoubleToString(call_wall, 2) + " (Resistance)\\n"
          + "Put Wall: " + DoubleToString(put_wall, 2) + " (Support)\\n"
          + "=====================");
}}

void OnTimer()
{{
    ReadGEXFile();
}}

int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
{{
    return(rates_total);
}}

void OnDeinit(const int reason)
{{
    ObjectDelete(0, "GEX_GammaFlip");
    ObjectDelete(0, "GEX_CallWall");
    ObjectDelete(0, "GEX_PutWall");
    Comment("");
}}
//+------------------------------------------------------------------+
'''

    file_path = os.path.join(indicators_path, "GEX_Levels.mq5")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(mql5_code)

    print(f"✅ MQL5 indicator saved: {file_path}")
    return True


# ============================================================
# 6. MAIN — Everything Connected
# ============================================================
def main():
    symbol = "PETR4"
    spot_price = 33.50

    print("=" * 50)
    print("  GEX → MetaTrader 5 Pipeline")
    print("=" * 50)

    # ── Step 1: Build options chain ──
    # Replace this with real data from B3/OpLab/Yahoo
    print("\n[1/5] Building options chain...")
    np.random.seed(42)
    strikes = np.arange(34, 44, 0.5)
    chain = pd.DataFrame({
        'strike': strikes,
        'call_oi': np.random.randint(1000, 50000, len(strikes)),
        'put_oi': np.random.randint(1000, 50000, len(strikes)),
        'call_iv': np.random.uniform(0.25, 0.55, len(strikes)),
        'put_iv': np.random.uniform(0.25, 0.55, len(strikes)),
        'expiration_date': datetime(2026, 4, 17),
    })
    print(f"      {len(chain)} strikes loaded ({strikes[0]} - {strikes[-1]})")

    # ── Step 2: Calculate GEX per strike ──
    print("\n[2/5] Calculating GEX per strike...")
    gex_df = calculate_gex_by_strike(chain, spot_price)
    print(f"      Calculated GEX for {len(gex_df)} strikes")

    # ── Step 3: Aggregate key levels ──
    print("\n[3/5] Aggregating GEX levels...")
    levels = aggregate_gex(gex_df)
    print(f"      Total GEX:   {levels['total_gex']:>12,.0f}")
    print(f"      Gamma Flip:  {levels['gamma_flip']}")
    print(f"      Call Wall:   {levels['call_wall']} (Resistance)")
    print(f"      Put Wall:    {levels['put_wall']} (Support)")
    print(f"      Regime:      {levels['regime']}")

    # ── Step 4: Write CSV/JSON to MT5 ── ★ THIS WAS MISSING ★
    print("\n[4/5] Sending GEX data to MT5...")
    file_ok = send_gex_to_mt5_via_file(levels, symbol)

    # ── Step 5: Install MQL5 indicator ──
    print("\n[5/5] Installing MQL5 indicator...")
    indicator_ok = install_mql5_indicator(symbol)

    # ── Summary ──
    print("\n" + "=" * 50)
    if file_ok and indicator_ok:
        print("  ✅ ALL DONE! Next steps:")
        print("     1. Open MetaEditor (F4)")
        print("     2. Navigate: Indicators/GEX/GEX_Levels.mq5")
        print("     3. Compile (F7)")
        print(f"     4. Drag indicator onto {symbol} chart")
        print("     5. Re-run this script to update GEX data")
    else:
        print("  ⚠️  MT5 not available. Files NOT written.")
        print("     Make sure MT5 is open and try again.")
    print("=" * 50)


if __name__ == "__main__":
    main()