"""
IBKRConnector
=============
Synchronous wrapper around the Interactive Brokers TWS / IB Gateway Python API.

Requirements
------------
    pip install ibapi

Setup
-----
1. Open TWS or IB Gateway.
2. Enable the Python API:
   TWS  → Edit → Global Configuration → API → Settings
          ✔ Enable ActiveX and Socket Clients
          Socket port: 7496 (live) / 7497 (paper)
   IB Gateway → Configure → API → Settings  (same options)
3. Adjust IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID in constants.py.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

from constants import IBKR_CLIENT_ID, IBKR_HOST, IBKR_PORT, IBKR_SYMBOL_CONFIG

_TIMEOUT = 15  # seconds to wait for each API response

# Mapping MT5 TIMEFRAME integer values → (IBKR bar-size string, duration unit)
# Duration unit: "S" = seconds, "D" = days, "W" = weeks
_TF_MAP: Dict[int, Tuple[str, str]] = {
    1:     ("1 min",   "S"),   # mt5.TIMEFRAME_M1
    5:     ("5 mins",  "S"),   # mt5.TIMEFRAME_M5
    15:    ("15 mins", "S"),   # mt5.TIMEFRAME_M15
    30:    ("30 mins", "S"),   # mt5.TIMEFRAME_M30
    16385: ("1 hour",  "H"),   # mt5.TIMEFRAME_H1
    16388: ("4 hours", "H"),   # mt5.TIMEFRAME_H4
    16408: ("1 day",   "D"),   # mt5.TIMEFRAME_D1
    32769: ("1 week",  "W"),   # mt5.TIMEFRAME_W1
}

# Seconds per bar (for computing IBKR "seconds" duration on intraday timeframes)
_TF_SECONDS: Dict[int, int] = {
    1: 60, 5: 300, 15: 900, 30: 1800, 16385: 3600, 16388: 14400,
}


# ── Internal EWrapper/EClient implementation ──────────────────────────────────

class _App(EWrapper, EClient):
    """Low-level combined EWrapper + EClient. Stores callback results keyed by reqId."""

    def __init__(self):
        EClient.__init__(self, self)
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._req_counter = 0

        self._bars:           Dict[int, list]            = {}
        self._bar_done:       Dict[int, threading.Event] = {}

        self._ticks:          Dict[int, dict]            = {}
        self._tick_done:      Dict[int, threading.Event] = {}

        self._opt_params:     Dict[int, list]            = {}
        self._opt_params_done: Dict[int, threading.Event] = {}

        self._cdetails:       Dict[int, list]            = {}
        self._cdetails_done:  Dict[int, threading.Event] = {}

    def _next(self) -> int:
        with self._lock:
            self._req_counter += 1
            return self._req_counter

    # ── Error handling ────────────────────────────────────────────────────────

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Informational codes — not real errors
        if errorCode in (2104, 2106, 2158, 2119):
            return
        self._logger.error("IBKR error reqId=%s code=%s: %s", reqId, errorCode, errorString)
        # Unblock any thread waiting on this reqId
        for store in (self._bar_done, self._tick_done, self._opt_params_done, self._cdetails_done):
            if reqId in store:
                store[reqId].set()

    # ── Historical data ───────────────────────────────────────────────────────

    def historicalData(self, reqId, bar):
        if reqId in self._bars:
            self._bars[reqId].append(bar)

    def historicalDataEnd(self, reqId, start, end):
        if reqId in self._bar_done:
            self._bar_done[reqId].set()

    # ── Market data snapshot ──────────────────────────────────────────────────

    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId not in self._ticks:
            self._ticks[reqId] = {}
        # tickType: 1=bid, 2=ask, 4=last
        label = {1: "bid", 2: "ask", 4: "last"}.get(tickType)
        if label and price > 0:
            self._ticks[reqId][label] = price

    def tickSnapshotEnd(self, reqId):
        if reqId in self._tick_done:
            self._tick_done[reqId].set()

    # ── Option parameters ─────────────────────────────────────────────────────

    def securityDefinitionOptionParameter(
        self, reqId, exchange, underlyingConId, tradingClass,
        multiplier, expirations, strikes,
    ):
        if reqId in self._opt_params:
            self._opt_params[reqId].append({
                "exchange":    exchange,
                "expirations": sorted(expirations),
                "strikes":     sorted(strikes),
                "multiplier":  multiplier,
            })

    def securityDefinitionOptionParameterEnd(self, reqId):
        if reqId in self._opt_params_done:
            self._opt_params_done[reqId].set()

    # ── Contract details ──────────────────────────────────────────────────────

    def contractDetails(self, reqId, contractDetails):
        if reqId in self._cdetails:
            self._cdetails[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        if reqId in self._cdetails_done:
            self._cdetails_done[reqId].set()


# ── Public connector ──────────────────────────────────────────────────────────

class IBKRConnector:
    """
    Synchronous market-data connector for Interactive Brokers TWS / IB Gateway.

    All public methods mirror the MT5Connector interface so that
    DataSourceService can delegate to this connector without changes to
    any existing call sites.
    """

    # Numeric value of mt5.TIMEFRAME_D1 — lets DataSourceService expose the
    # same class-level constant regardless of which connector is active.
    TIMEFRAME_D1 = 16408

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._app = _App()
        self._app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

        self._thread = threading.Thread(
            target=self._app.run, daemon=True, name="ibkr-msg-loop"
        )
        self._thread.start()

        # Wait for the socket handshake to complete
        deadline = time.time() + _TIMEOUT
        while not self._app.isConnected() and time.time() < deadline:
            time.sleep(0.05)

        if not self._app.isConnected():
            raise RuntimeError(
                f"Could not connect to IBKR TWS/Gateway at {IBKR_HOST}:{IBKR_PORT}. "
                "Make sure TWS or IB Gateway is open with the API enabled."
            )

        self._logger.info("IBKRConnector connected to %s:%s (client %s)", IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

    # ── Contract factory helpers ──────────────────────────────────────────────

    def _contract_spec(self, symbol: str) -> tuple:
        """Return (secType, exchange, currency) from IBKR_SYMBOL_CONFIG, with BVMF/STK/BRL as defaults."""
        cfg = IBKR_SYMBOL_CONFIG.get(symbol, {})
        return (
            cfg.get("secType", "STK"),
            cfg.get("exchange", "BVMF"),
            cfg.get("currency", "BRL"),
        )

    def _underlying_contract(self, symbol: str) -> Contract:
        """Build the correct underlying Contract for a symbol using IBKR_SYMBOL_CONFIG."""
        sec_type, exchange, currency = self._contract_spec(symbol)
        c = Contract()
        c.symbol   = symbol
        c.secType  = sec_type
        c.exchange = exchange
        c.currency = currency
        return c

    @staticmethod
    def _stock(symbol: str, exchange: str = "BVMF", currency: str = "BRL") -> Contract:
        c = Contract()
        c.symbol   = symbol
        c.secType  = "STK"
        c.exchange = exchange
        c.currency = currency
        return c

    @staticmethod
    def _future(symbol: str, exchange: str = "BVMF", currency: str = "BRL") -> Contract:
        c = Contract()
        c.symbol   = symbol
        c.secType  = "FUT"
        c.exchange = exchange
        c.currency = currency
        return c

    # ── Internal: contract details ────────────────────────────────────────────

    def _req_contract_details(self, contract: Contract) -> list:
        rid = self._app._next()
        ev  = threading.Event()
        self._app._cdetails[rid]      = []
        self._app._cdetails_done[rid] = ev
        self._app.reqContractDetails(rid, contract)
        ev.wait(_TIMEOUT)
        details = self._app._cdetails.pop(rid, [])
        self._app._cdetails_done.pop(rid, None)
        return details

    # ── Historical OHLCV ─────────────────────────────────────────────────────

    def get_data(
        self, symbol: str, timeframe: int, periods: int, shift: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars from IBKR.
        Parameters mirror MT5Connector.get_data() so callers need no changes.
        """
        tf = _TF_MAP.get(timeframe)
        if tf is None:
            self._logger.error("Unsupported timeframe value for IBKR: %s", timeframe)
            return None

        bar_size, unit = tf
        total = periods + shift

        if unit == "D":
            duration = f"{total} D"
        elif unit == "W":
            duration = f"{total} W"
        else:
            # Intraday: express duration in seconds
            secs = _TF_SECONDS.get(timeframe, 60) * total
            duration = f"{secs} S"

        rid = self._app._next()
        ev  = threading.Event()
        self._app._bars[rid]     = []
        self._app._bar_done[rid] = ev

        self._app.reqHistoricalData(
            rid, self._stock(symbol),
            "",           # endDateTime: "" = now
            duration,
            bar_size,
            "TRADES",
            1,            # useRTH: regular trading hours only
            1,            # formatDate: "YYYYMMDD HH:MM:SS"
            False,        # keepUpToDate
            [],
        )

        if not ev.wait(_TIMEOUT):
            self._logger.error("Timeout fetching historical data for %s", symbol)
            self._app._bars.pop(rid, None)
            self._app._bar_done.pop(rid, None)
            return None

        bars = self._app._bars.pop(rid, [])
        self._app._bar_done.pop(rid, None)

        if not bars:
            return None

        rows = []
        for b in bars:
            try:
                dt = (
                    datetime.strptime(b.date, "%Y%m%d %H:%M:%S")
                    if " " in b.date
                    else datetime.strptime(b.date, "%Y%m%d")
                )
            except ValueError:
                continue
            rows.append({
                "time": dt, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "tick_volume": b.volume,
            })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df = df[df["time"].dt.weekday < 5]   # strip weekends

        if shift > 0:
            df = df.iloc[:-shift] if len(df) > shift else df.iloc[0:0]

        return df.tail(periods).reset_index(drop=True)

    # ── Real-time quote ───────────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Fetch a real-time snapshot quote from IBKR.
        Returns a dict with keys: bid, ask, last.
        Returns None when the quote cannot be obtained.
        """
        rid = self._app._next()
        ev  = threading.Event()
        self._app._ticks[rid]     = {}
        self._app._tick_done[rid] = ev

        self._app.reqMktData(rid, self._underlying_contract(symbol), "", True, False, [])
        ev.wait(_TIMEOUT)
        self._app.cancelMktData(rid)

        ticks = self._app._ticks.pop(rid, {})
        self._app._tick_done.pop(rid, None)

        if not ticks:
            return None

        return {
            "bid":  ticks.get("bid",  0.0),
            "ask":  ticks.get("ask",  0.0),
            "last": ticks.get("last", 0.0),
        }

    # ── Options chain parameters ──────────────────────────────────────────────

    def get_option_params(
        self, symbol: str, exchange: str = None
    ) -> List[dict]:
        """
        Retrieve available option expirations and strikes for an underlying.
        Returns a list of dicts: {exchange, expirations, strikes, multiplier}
        where expirations is a sorted list of 'YYYYMMDD' strings.
        """
        sec_type, sym_exchange, _currency = self._contract_spec(symbol)
        details = self._req_contract_details(self._underlying_contract(symbol))
        if not details:
            self._logger.error("Could not resolve contract details for %s", symbol)
            return []

        con_id = details[0].contract.conId
        rid = self._app._next()
        ev  = threading.Event()
        self._app._opt_params[rid]       = []
        self._app._opt_params_done[rid]  = ev

        # futFopExchange is only relevant for FUT underlyings; leave blank for STK/IND.
        fut_fop_exchange = (exchange or sym_exchange) if sec_type == "FUT" else ""
        self._app.reqSecDefOptParams(rid, symbol, fut_fop_exchange, sec_type, con_id)
        ev.wait(_TIMEOUT)

        params = self._app._opt_params.pop(rid, [])
        self._app._opt_params_done.pop(rid, None)
        return params

    # ── Futures (DI1 / interest rate curve) ──────────────────────────────────

    def get_symbol_futures(
        self, symbol: str, exchange: str = "BVMF"
    ) -> Optional[tuple]:
        """
        Return the nearest active futures contract as (expiry_timestamp, symbol_name).
        Mirrors MT5Connector.get_symbol_futures() return value.
        """
        details = self._req_contract_details(self._future(symbol, exchange=exchange))
        if not details:
            self._logger.error("No futures contracts found for %s", symbol)
            return None

        now_ts = time.time()

        def _ts(d) -> float:
            raw = d.contract.lastTradeDateOrContractMonth
            try:
                return datetime.strptime(raw[:8], "%Y%m%d").timestamp()
            except ValueError:
                return 0.0

        nearest = min(
            (d for d in details if _ts(d) > now_ts),
            key=_ts,
            default=None,
        )
        if nearest is None:
            return None

        return (int(_ts(nearest)), nearest.contract.localSymbol or nearest.contract.symbol)
    # ── Option names by expiration (mirrors MT5Connector) ──────────────────────

    def get_option_names_by_expiration_time(
        self,
        symbol: str,
        expiry_rank: int = 1,
        exchange: str = None,
    ) -> dict:
        """
        Return ``{expiry_timestamp: [option_local_symbols]}`` for the chosen
        expiry rank, mirroring MT5Connector.get_option_names_by_expiration_time().

        Steps:
          1. Call reqSecDefOptParams to discover available expiration dates.
          2. Pick the Nth expiration (expiry_rank, 1-based).
          3. Call reqContractDetails for all options at that expiry to collect
             individual contract local symbols.
        """
        # Step 1: get available expirations
        params_list = self.get_option_params(symbol)
        if not params_list:
            self._logger.error("No option params returned for %s from IBKR", symbol)
            return {}

        _, sym_exchange, _c = self._contract_spec(symbol)
        resolved_exchange = exchange or sym_exchange

        # Prefer the entry matching the resolved exchange; fall back to first with data
        expirations: list = []
        for entry in params_list:
            if entry["exchange"] == resolved_exchange and entry["expirations"]:
                expirations = entry["expirations"]
                break
        if not expirations:
            for entry in params_list:
                if entry["expirations"]:
                    expirations = entry["expirations"]
                    break

        if not expirations:
            self._logger.error("No expirations found for %s on %s", symbol, exchange)
            return {}

        idx = min(max(expiry_rank - 1, 0), len(expirations) - 1)
        chosen_expiry_str = expirations[idx]  # 'YYYYMMDD'
        chosen_ts = int(datetime.strptime(chosen_expiry_str, "%Y%m%d").timestamp())

        # Step 2: request all option contracts for this expiry
        _sec_type, sym_exchange, sym_currency = self._contract_spec(symbol)
        opt_contract = Contract()
        opt_contract.symbol                         = symbol
        opt_contract.secType                        = "OPT"
        opt_contract.exchange                       = exchange or sym_exchange
        opt_contract.currency                       = sym_currency
        opt_contract.lastTradeDateOrContractMonth   = chosen_expiry_str

        details = self._req_contract_details(opt_contract)
        if not details:
            self._logger.warning(
                "No option contracts found for %s expiry %s on IBKR",
                symbol, chosen_expiry_str,
            )
            return {chosen_ts: []}

        names = [
            d.contract.localSymbol or
            f"{symbol}{chosen_expiry_str}{d.contract.right}{d.contract.strike:.0f}"
            for d in details
        ]
        self._logger.info(
            "IBKR option chain for %s expiry %s: %d contracts",
            symbol, chosen_expiry_str, len(names),
        )
        return {chosen_ts: names}
    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def disconnect(self):
        self._app.disconnect()
        self._logger.info("IBKRConnector disconnected.")
