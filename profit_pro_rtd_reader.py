"""
Read Open Interest and Options data from Profit Pro via RTD (Real-Time Data).

RTD is superior to DDE:
  - More stable connection
  - Automatic updates (push, not poll)
  - Better error handling
  - Native Windows COM integration

Requirements:
  pip install pywin32

Usage:
  1. Open Profit Pro
  2. Run this script
  3. Data flows automatically
"""

import pythoncom
import win32com.client
import time
import pandas as pd
from datetime import datetime
from typing import Optional


class ProfitProRTD:
    """
    Connect to Profit Pro via RTD (Real-Time Data) protocol.

    RTD Server Name (Nelogica):
        "RTDProfitChart.RTDServer"
        or
        "ProfitChart.RtdServer.1"

    NOTE: The exact ProgID may vary by Profit Pro version.
    Check in: Profit Pro → Ferramentas → Configurações → RTD/DDE
    """

    # Common RTD server ProgIDs for Profit Pro (try each)
    RTD_SERVER_IDS = [
        "RTDProfitChart.RTDServer",
        "ProfitChart.RtdServer.1",
        "ProfitPro.RtdServer",
        "Nelogica.ProfitChart.RTD",
    ]

    def __init__(self, server_prog_id: str = None):
        self.rtd = None
        self.server_prog_id = server_prog_id
        self.connected = False

    def connect(self) -> bool:
        """
        Initialize RTD COM connection to Profit Pro.
        Tries multiple known ProgIDs if none specified.
        """
        pythoncom.CoInitialize()

        prog_ids = (
            [self.server_prog_id] if self.server_prog_id
            else self.RTD_SERVER_IDS
        )

        for prog_id in prog_ids:
            try:
                self.rtd = win32com.client.Dispatch(prog_id)
                # RTD servers require ServerStart
                result = self.rtd.ServerStart(None)
                if result > 0:
                    raise Exception(f"ServerStart returned error: {result}")

                self.server_prog_id = prog_id
                self.connected = True
                print(f"✅ Connected to Profit Pro RTD: {prog_id}")
                return True

            except Exception as e:
                print(f"   ⚠️  {prog_id} → {e}")
                continue

        print("\n❌ Could not connect to any RTD server.")
        print("   Possible fixes:")
        print("   1. Make sure Profit Pro is OPEN")
        print("   2. Check RTD ProgID in: Profit → Tools → Settings → RTD")
        print("   3. Run Python as Administrator")
        print("   4. Register COM: regsvr32 ProfitChartRTD.dll")
        self.connected = False
        return False

    def get_value(self, topic1: str, topic2: str = "") -> Optional[str]:
        """
        Request a single RTD value.

        RTD works with "topics" — each topic identifies a data field.
        Profit Pro typically uses:
            Topic1 = Asset code (e.g., "PETRD370")
            Topic2 = Field name (e.g., "ABE", "ULT", "VOL", "OI")
        """
        if not self.connected:
            print("❌ Not connected. Call connect() first.")
            return None

        try:
            topics = (topic1, topic2) if topic2 else (topic1,)
            # ConnectData returns a topic_id
            topic_id = id(topics)  # unique identifier
            result = self.rtd.ConnectData(topic_id, topics, True)

            # RefreshData retrieves updated values
            _, values = self.rtd.RefreshData(1000)
            if values:
                return str(values)

            return str(result)

        except Exception as e:
            print(f"❌ RTD request failed [{topic1}|{topic2}]: {e}")
            return None

    def disconnect(self):
        """Shutdown RTD connection."""
        if self.rtd:
            try:
                self.rtd.ServerTerminate()
            except Exception:
                pass
        pythoncom.CoUninitialize()
        self.connected = False
        print("🔌 RTD disconnected")


class ProfitProRTDExcel:
    """
    Alternative: Use RTD through Excel COM automation.
    More reliable because Excel handles the RTD protocol natively.

    Flow: Python → Excel → RTD → Profit Pro
    """

    def __init__(self, rtd_prog_id: str = "RTDProfitChart.RTDServer"):
        self.excel = None
        self.workbook = None
        self.sheet = None
        self.rtd_prog_id = rtd_prog_id

    def connect(self) -> bool:
        """Open Excel and prepare RTD formulas."""
        try:
            pythoncom.CoInitialize()
            self.excel = win32com.client.Dispatch("Excel.Application")
            self.excel.Visible = False  # Hidden Excel
            self.workbook = self.excel.Workbooks.Add()
            self.sheet = self.workbook.Sheets(1)
            print("✅ Excel COM initialized for RTD bridge")
            return True
        except Exception as e:
            print(f"❌ Excel initialization failed: {e}")
            return False

    def set_rtd_cell(self, cell: str, asset: str, field: str):
        """
        Insert an RTD formula into an Excel cell.

        Equivalent to typing in Excel:
        =RTD("RTDProfitChart.RTDServer",,"PETRD370","ULT")
        """
        formula = f'=RTD("{self.rtd_prog_id}",,"{asset}","{field}")'
        self.sheet.Range(cell).Formula = formula

    def get_cell_value(self, cell: str):
        """Read the calculated value from an Excel cell."""
        # Give Excel time to calculate RTD
        time.sleep(0.5)
        return self.sheet.Range(cell).Value

    def build_options_chain(
        self,
        underlying: str = "PETR4",
        series: str = "D",
        strikes: list = None,
    ) -> pd.DataFrame:
        """
        Build a complete options chain via RTD through Excel.

        Parameters:
            underlying: "PETR4", "VALE3", etc.
            series: Option series letter (A-L calls, M-X puts)
                    A/M=Jan, B/N=Feb, C/O=Mar, D/P=Apr ...
            strikes: List of strike prices [34, 35, 36, ...]
        """
        if strikes is None:
            strikes = list(range(30, 46))

        prefix = underlying[:4]  # PETR
        call_series = series.upper()
        # Put series = Call series + 12 letters ahead
        put_series = chr(ord(call_series) + 12)  # D → P

        print(f"\n📡 Loading options chain via RTD...")
        print(f"   Underlying: {underlying}")
        print(f"   Call series: {call_series} | Put series: {put_series}")

        # RTD fields we want
        fields = {
            'ULT': 'last_price',    # Último preço
            'VOL': 'volume',        # Volume
            'ABE': 'open',          # Abertura
            'MAX': 'high',          # Máxima
            'MIN': 'low',           # Mínima
        }

        # OI field name — varies by Profit version
        oi_fields = ['CTAB', 'OI', 'OIAB', 'CONTAB']  # try these

        chain_data = []
        row = 1  # Excel row counter

        for strike in strikes:
            # Strike formatting: 370 = 37.00, 3450 = 34.50
            strike_code = str(int(strike * 100)).rjust(3, '0') if strike < 100 else str(int(strike))

            call_code = f"{prefix}{call_series}{strike_code}"
            put_code = f"{prefix}{put_series}{strike_code}"

            record = {'strike': float(strike)}

            # ── Call data ──
            for field_key, field_name in fields.items():
                cell = f"A{row}"
                self.set_rtd_cell(cell, call_code, field_key)
                row += 1

            # Call OI
            cell_oi_call = f"A{row}"
            self.set_rtd_cell(cell_oi_call, call_code, "CTAB")
            row += 1

            # ── Put data ──
            for field_key, field_name in fields.items():
                cell = f"A{row}"
                self.set_rtd_cell(cell, put_code, field_key)
                row += 1

            # Put OI
            cell_oi_put = f"A{row}"
            self.set_rtd_cell(cell_oi_put, put_code, "CTAB")
            row += 1

        # Wait for RTD to populate
        print("   ⏳ Waiting for RTD data (5 seconds)...")
        time.sleep(5)

        # Now read all values back
        row = 1
        for strike in strikes:
            record = {'strike': float(strike)}

            # Read call fields
            for field_name in fields.values():
                val = self.get_cell_value(f"A{row}")
                record[f'call_{field_name}'] = self._to_float(val)
                row += 1

            record['call_oi'] = self._to_float(self.get_cell_value(f"A{row}"))
            row += 1

            # Read put fields
            for field_name in fields.values():
                val = self.get_cell_value(f"A{row}")
                record[f'put_{field_name}'] = self._to_float(val)
                row += 1

            record['put_oi'] = self._to_float(self.get_cell_value(f"A{row}"))
            row += 1

            chain_data.append(record)
            print(f"   Strike {strike}: Call OI={record['call_oi']:.0f} | Put OI={record['put_oi']:.0f}")

        df = pd.DataFrame(chain_data)
        print(f"\n✅ Options chain loaded: {len(df)} strikes")
        return df

    def _to_float(self, value) -> float:
        """Safely convert RTD value to float."""
        if value is None:
            return 0.0
        try:
            if isinstance(value, str):
                value = value.replace(',', '.').strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def disconnect(self):
        """Close Excel."""
        try:
            if self.workbook:
                self.workbook.Close(SaveChanges=False)
            if self.excel:
                self.excel.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()
        print("🔌 Excel RTD bridge closed")


# ============================================================
# Integration with GEX Pipeline
# ============================================================
def fetch_oi_from_profit_rtd(
    underlying: str = "PETR4",
    series: str = "D",
    spot_price: float = 38.50,
    strikes: list = None,
) -> pd.DataFrame:
    """
    Complete function: Fetch OI from Profit Pro via RTD
    and return a DataFrame ready for GEX calculation.

    Parameters:
        underlying: Asset code (PETR4, VALE3, BOVA11, etc.)
        series:     Option series letter (D = April Calls)
        spot_price: Current price of underlying
        strikes:    List of strikes to query

    Returns:
        DataFrame with columns needed for GEX:
        [strike, call_oi, put_oi, call_iv, put_iv, expiration_date]
    """

    if strikes is None:
        # Generate strikes around spot price
        base = int(spot_price)
        strikes = [base + i * 0.5 for i in range(-10, 11)]

    rtd = ProfitProRTDExcel()

    if not rtd.connect():
        return pd.DataFrame()

    try:
        chain = rtd.build_options_chain(underlying, series, strikes)

        if chain.empty:
            return pd.DataFrame()

        # Add estimated IV (use BS inversion in production)
        chain['call_iv'] = 0.35  # placeholder
        chain['put_iv'] = 0.35   # placeholder

        # Add expiration (map series letter to month)
        series_to_month = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
            'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
        }
        month = series_to_month.get(series.upper(), 4)
        year = datetime.now().year
        # B3 options expire on 3rd Monday of the month (approximately)
        chain['expiration_date'] = datetime(year, month, 17)

        # Select only columns needed for GEX
        result = chain[['strike', 'call_oi', 'put_oi', 'call_iv', 'put_iv', 'expiration_date']].copy()

        return result

    finally:
        rtd.disconnect()


# ============================================================
# MAIN — Full Pipeline: Profit RTD → GEX → MT5
# ============================================================
if __name__ == "__main__":

    print("=" * 55)
    print("  Profit Pro RTD → GEX → MetaTrader 5")
    print("=" * 55)

    SYMBOL = "PETR4"
    SPOT = 38.50
    SERIES = "D"  # April

    # ── Step 1: Fetch OI from Profit Pro ──
    print("\n[1/4] Fetching OI from Profit Pro via RTD...")
    chain = fetch_oi_from_profit_rtd(
        underlying=SYMBOL,
        series=SERIES,
        spot_price=SPOT,
    )

    if chain.empty:
        print("\n❌ Could not fetch data from Profit Pro.")
        print("   Falling back to sample data for testing...\n")

        import numpy as np
        np.random.seed(42)
        strikes = [SPOT + i * 0.5 for i in range(-10, 11)]
        chain = pd.DataFrame({
            'strike': strikes,
            'call_oi': np.random.randint(1000, 50000, len(strikes)),
            'put_oi': np.random.randint(1000, 50000, len(strikes)),
            'call_iv': np.random.uniform(0.25, 0.55, len(strikes)),
            'put_iv': np.random.uniform(0.25, 0.55, len(strikes)),
            'expiration_date': datetime(2026, 4, 17),
        })

    print(f"   ✅ Chain loaded: {len(chain)} strikes")
    print(chain[['strike', 'call_oi', 'put_oi']].to_string(index=False))

    # ── Step 2: Calculate GEX ──
    # Import from the complete pipeline
    try:
        from gex_to_mt5 import (
            calculate_gex_by_strike,
            aggregate_gex,
            send_gex_to_mt5_via_file,
            install_mql5_indicator,
        )

        print(f"\n[2/4] Calculating GEX...")
        gex_df = calculate_gex_by_strike(chain, SPOT)

        print(f"\n[3/4] Aggregating levels...")
        levels = aggregate_gex(gex_df)
        print(f"   Total GEX:   {levels['total_gex']:>12,.0f}")
        print(f"   Gamma Flip:  {levels['gamma_flip']}")
        print(f"   Call Wall:   {levels['call_wall']} (Resistance)")
        print(f"   Put Wall:    {levels['put_wall']} (Support)")
        print(f"   Regime:      {levels['regime']}")

        print(f"\n[4/4] Sending to MT5...")
        send_gex_to_mt5_via_file(levels, SYMBOL)
        install_mql5_indicator(SYMBOL)

        print("\n" + "=" * 55)
        print("  ✅ Pipeline complete!")
        print("=" * 55)

    except ImportError:
        print("\n⚠️  gex_to_mt5.py not found.")
        print("   Place both files in the same directory.")
        print("   Chain data was loaded successfully — GEX calculation skipped.")