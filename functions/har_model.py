#!/usr/bin/env python3
"""
HAR (Heterogeneous Autoregressive) volatility forecast module.

Provides a single entry-point function that takes daily close prices
and returns an annualized volatility forecast comparable to GARCH output.

Based on Corsi (2009) HAR-RV model using log(RV) regression with
daily(1), weekly(5) and monthly(22) averaged realized-variance levels.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def _daily_rv_from_close(prices: pd.Series) -> pd.Series:
    lnret = np.log(prices).diff()
    rv = lnret.pow(2).rename("RV")
    return rv.dropna()


def _build_har_features(rv: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"RV": rv})
    df["RV"] = df["RV"].replace(0, np.nan).fillna(df["RV"][df["RV"] > 0].min() * 0.01)
    df["RV"] = df["RV"].clip(lower=1e-10)

    df["d"] = df["RV"]
    df["w"] = df["RV"].rolling(window=5).mean()
    df["m"] = df["RV"].rolling(window=22).mean()
    df["y_fwd"] = np.log(df["RV"].shift(-1))
    df["log_d"] = np.log(df["d"])
    df["log_w"] = np.log(df["w"])
    df["log_m"] = np.log(df["m"])
    df = df.dropna().copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _fit_har_logrv(df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    X = np.column_stack([np.ones(len(df)), df[["log_d", "log_w", "log_m"]].to_numpy()])
    y = df["y_fwd"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res_var = float(np.mean((y - X @ beta) ** 2))
    return beta, res_var


def har_annual_vol_forecast(close_prices: pd.Series, trading_days: int = 252) -> Tuple[float, float]:
    """
    Compute HAR one-step-ahead annualized volatility forecast from daily close prices.

    Args:
        close_prices: pandas Series of daily close prices.
        trading_days: number of trading days per year (default 252).

    Returns:
        (annualized_vol, daily_RV_forecast)

    Raises:
        ValueError: if input is invalid or too short for HAR estimation.
    """
    if not isinstance(close_prices, pd.Series):
        raise ValueError("close_prices must be a pandas Series")

    rv = _daily_rv_from_close(close_prices)
    df = _build_har_features(rv)
    if df.shape[0] < 60:
        raise ValueError("Not enough data after rolling windows to fit HAR (need ~60+ rows).")

    beta, res_var = _fit_har_logrv(df)
    last_row = df.iloc[-1]
    X_last = np.array([1.0, last_row["log_d"], last_row["log_w"], last_row["log_m"]])
    yhat_log = float(X_last @ beta)
    f_RV = float(np.exp(yhat_log + 0.5 * res_var))
    vol_annual = float(np.sqrt(f_RV * trading_days))
    return vol_annual, f_RV
