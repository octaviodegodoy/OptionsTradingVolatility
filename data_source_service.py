"""
DataSourceService
=================
Routes market-data operations to the platform configured in constants.py.

    ACTIVE_DATA_SOURCE = DATA_SOURCE_MT5   → use MetaTrader 5
    ACTIVE_DATA_SOURCE = DATA_SOURCE_IBKR  → use Interactive Brokers TWS API

Rules
-----
* Data (OHLCV, quotes, option params, futures) → configured source only.
* Options chain discovery (names/expiries)      → MT5 broker feed always.
* Execution (orders, positions, account)        → MT5 always.
* No silent fallback: if the configured source fails, an error is raised.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from constants import (
    ACTIVE_DATA_SOURCE,
    DATA_SOURCE_IBKR,
    DATA_SOURCE_MT5,
)
from mt5_connector import MT5Connector


# ── Normalised quote DTO ──────────────────────────────────────────────────────

@dataclass
class SymbolQuote:
    """
    Platform-agnostic quote returned by get_symbol_info() when IBKR is active.
    Exposes the same attribute names as MT5 SymbolInfo so callers need no changes.
    """
    name: str
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    option_strike: float = 0.0
    filling_mode: int = 1


# ── Service ───────────────────────────────────────────────────────────────────

class DataSourceService:
    """
    Single entry-point for all data and trading operations.

    Replace ``MT5Connector()`` with ``DataSourceService()`` in any call site.
    The public API is identical.  To switch platform edit constants.py::

        ACTIVE_DATA_SOURCE = DATA_SOURCE_IBKR
    """

    ORDER_TYPE_BUY  = MT5Connector.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = MT5Connector.ORDER_TYPE_SELL
    TIMEFRAME_D1    = MT5Connector.TIMEFRAME_D1

    def __init__(self):
        self._source = ACTIVE_DATA_SOURCE
        self._logger = logging.getLogger(__name__)

        if self._source not in (DATA_SOURCE_MT5, DATA_SOURCE_IBKR):
            raise ValueError(
                f"Unsupported ACTIVE_DATA_SOURCE: {self._source!r}. "
                f"Must be {DATA_SOURCE_MT5!r} or {DATA_SOURCE_IBKR!r}."
            )

        # MT5 is always needed for execution (orders/positions/account).
        self._mt5 = MT5Connector()

        # Data connector — MT5 or IBKR, no mixing.
        if self._source == DATA_SOURCE_IBKR:
            from ibkr_connector import IBKRConnector
            self._ibkr = IBKRConnector()   # raises RuntimeError if TWS unreachable
        else:
            self._ibkr = None

        self._logger.info("DataSourceService ready. Active source: %s", self._source)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def active_source(self) -> str:
        """The currently configured data source identifier string."""
        return self._source

    def _data(self):
        """Return the active data connector."""
        return self._ibkr if self._source == DATA_SOURCE_IBKR else self._mt5

    # ── Historical OHLCV ─────────────────────────────────────────────────────

    def get_data(self, symbol, timeframe, periods, shift):
        return self._data().get_data(symbol, timeframe, periods, shift)

    # ── Real-time quote / symbol info ─────────────────────────────────────────

    def get_symbol_info(self, symbol):
        if self._source == DATA_SOURCE_IBKR:
            raw = self._ibkr.get_symbol_info(symbol)
            if raw is None:
                return None
            return SymbolQuote(
                name=symbol,
                bid=raw["bid"],
                ask=raw["ask"],
                last=raw.get("last", 0.0),
            )
        return self._mt5.get_symbol_info(symbol)

    # ── Symbol selection (MT5 Market Watch) ──────────────────────────────────

    def symbol_select(self, symbol, enable):
        return self._mt5.symbol_select(symbol, enable)

    def unselect_options_by_underlying(self, underlying: str) -> int:
        return self._mt5.unselect_options_by_underlying(underlying)

    def select_options_near_spot(
        self,
        underlying: str,
        expiry_rank: int = 1,
        strike_pct: float = 0.10,
    ) -> int:
        return self._mt5.select_options_near_spot(underlying, expiry_rank, strike_pct)

    def unselect_bova_options(self, symbol: str = "BOVA11") -> int:
        return self._mt5.unselect_bova_options(symbol)

    # ── Option parameters (expirations + strikes) ─────────────────────────────

    def get_option_params(self, symbol: str, exchange: str = "BVMF") -> list:
        """
        Return available option expirations and strikes for *symbol*.

        IBKR returns a list of dicts::

            [{
                'exchange':    str,
                'expirations': ['YYYYMMDD', ...],
                'strikes':     [float, ...],
                'multiplier':  str,
            }]

        MT5 returns the raw chain dict from get_option_names_by_expiration_time.
        """
        if self._source == DATA_SOURCE_IBKR:
            return self._ibkr.get_option_params(symbol, exchange)
        # MT5: wrap chain dict in the same shape for uniform handling
        chain = self._mt5.get_option_names_by_expiration_time(symbol)
        if not chain:
            return []
        from datetime import datetime as _dt
        result = []
        for expiry_ts, names in chain.items():
            result.append({
                "exchange":    "MT5",
                "expirations": [_dt.fromtimestamp(expiry_ts).strftime("%Y%m%d")],
                "strikes":     [],
                "names":       names,
                "multiplier":  "1",
            })
        return result

    # ── Options chain — MT5 broker feed ──────────────────────────────────────

    def get_options_chain(self, group_name, option_type):
        return self._mt5.get_options_chain(group_name, option_type)

    def get_option_names_by_expiration_time(
        self,
        symbol,
        expiry_rank_override: int = None,
    ):
        return self._mt5.get_option_names_by_expiration_time(symbol, expiry_rank_override)

    def get_option_name_by_strike(
        self,
        group_name,
        strike_price,
        option_type,
        expiration_time,
    ):
        return self._mt5.get_option_name_by_strike(
            group_name, strike_price, option_type, expiration_time
        )

    def get_call_option_name_list(self, group_name):
        return self._mt5.get_call_option_name_list(group_name)

    def get_put_option_name_list(self, symbol):
        return self._mt5.get_put_option_name_list(symbol)

    # ── Futures / DI1 ────────────────────────────────────────────────────────

    def get_symbol_futures(self, group_name):
        return self._data().get_symbol_futures(group_name)

    # ── Execution — always MT5 ────────────────────────────────────────────────

    def place_order(self, symbol, order_type, volume, price, deviation, comment):
        return self._mt5.place_order(symbol, order_type, volume, price, deviation, comment)

    def place_order_vertical(self, symbolY, symbolX, orders_type, volume, iv_y, iv_x):
        return self._mt5.place_order_vertical(
            symbolY, symbolX, orders_type, volume, iv_y, iv_x
        )

    def close_all_positions(self):
        return self._mt5.close_all_positions()

    # ── Account / position info — always MT5 ─────────────────────────────────

    def get_account_info(self):
        return self._mt5.get_account_info()

    def get_open_positions(self):
        return self._mt5.get_open_positions()

    def get_total_volume(self):
        return self._mt5.get_total_volume()

    def get_total_positions(self):
        return self._mt5.get_total_positions()

    def get_profit(self):
        return self._mt5.get_profit()

    def total_daily_risk(self):
        return self._mt5.total_daily_risk()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self):
        return self._mt5.initialize()

    def shutdown(self):
        if self._ibkr is not None:
            self._ibkr.disconnect()
            self._ibkr = None
        self._mt5.shutdown()

    def sleep(self, seconds):
        time.sleep(seconds)

    # ── Misc ──────────────────────────────────────────────────────────────────

    def last_error(self):
        return self._mt5.last_error()

    def get_mt5_connector(self):
        """Direct access to the raw MT5 module (for advanced/legacy callers)."""
        return self._mt5.get_mt5_connector()
