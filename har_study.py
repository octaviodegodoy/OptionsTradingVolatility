#!/usr/bin/env python3
"""
HAR-based annual volatility forecast from a price series.

Usage:
- Provide a pandas Series of CLOSE prices indexed by datetime (daily frequency).
- By default it treats daily squared log-returns as realized variance (RV).
- Fit HAR on log(RV) with daily(1), weekly(5) and monthly(22) averaged levels.
- Forecast next-day RV, bias-correct the log back-transform, then annualize:
    vol_annual = sqrt(forecast_daily_RV * trading_days)

Notes:
- For more accurate RV use high-frequency returns to compute intraday-sum RV and feed that series.
- trading_days default 252; change to your market convention.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

from mt5_connector import MT5Connector


def daily_rv_from_close(prices: pd.Series) -> pd.Series:
    """
    Simple realized variance proxy from daily close prices:
      returns = ln(P_t / P_{t-1})
      RV_t = returns**2  (daily realized variance)
    If you have intraday returns, replace with sum(intraday_returns**2) per day.
    """
    lnret = np.log(prices).diff()
    rv = lnret.pow(2).rename("RV")
    return rv.dropna()


def build_har_features(rv: pd.Series) -> pd.DataFrame:
    """
    Build HAR features (levels) and log-target (log RV_{t+1}).
    Returns DataFrame aligned so each row t has predictors computed at t
    and target 'y_fwd' = log(RV_{t+1}).
    """
    df = pd.DataFrame({"RV": rv})
    # Replace zeros and negative values with small positive number to avoid log issues
    df["RV"] = df["RV"].replace(0, np.nan).fillna(df["RV"][df["RV"] > 0].min() * 0.01)
    df["RV"] = df["RV"].clip(lower=1e-10)
    
    df["d"] = df["RV"]                       # daily level
    df["w"] = df["RV"].rolling(window=5).mean()   # weekly avg of levels
    df["m"] = df["RV"].rolling(window=22).mean()  # monthly avg of levels
    df["y_fwd"] = np.log(df["RV"].shift(-1))     # target: log RV_{t+1}
    # Work in logs for regression: predictors are log(levels)
    df["log_d"] = np.log(df["d"])
    df["log_w"] = np.log(df["w"])
    df["log_m"] = np.log(df["m"])
    df = df.dropna().copy()
    # Remove any remaining infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def fit_har_logrv(df: pd.DataFrame) -> Tuple[object, float]:
    """
    Fit OLS on log(RV_{t+1}) ~ 1 + log_d + log_w + log_m
    Returns (fitted_model, in_sample_residual_variance)
    """
    X = sm.add_constant(df[["log_d", "log_w", "log_m"]])
    y = df["y_fwd"]
    model = sm.OLS(y, X).fit()
    res_var = float(np.mean(model.resid ** 2))  # residual variance in log space
    return model, res_var


def forecast_next_daily_rv(model, res_var: float, last_rv_row: pd.Series) -> float:
    """
    last_rv_row must contain columns: log_d, log_w, log_m (computed at last available day t).
    Returns bias-corrected forecast of RV_{t+1} (level, not log).
    """
    X_last = np.array([1.0, last_rv_row["log_d"], last_rv_row["log_w"], last_rv_row["log_m"]])
    yhat_log = float(np.dot(X_last, model.params.values))
    # Bias correction for log-normal error
    f_RV = float(np.exp(yhat_log + 0.5 * res_var))
    return f_RV


def annualize_vol_from_daily_rv(daily_rv: float, trading_days: int = 252) -> float:
    """
    Convert daily realized variance to annualized volatility (std dev).
    vol_annual = sqrt( daily_rv * trading_days )
    """
    return float(np.sqrt(daily_rv * trading_days))


# --------- Example end-to-end function ----------
def har_annual_vol_forecast_from_prices(
    close_prices: pd.Series,
    trading_days: int = 252
) -> Tuple[float, float, object]:
    """
    Compute HAR one-step-ahead forecast and return:
      (annualized_vol, daily_RV_forecast, fitted_model)
    """
    if not isinstance(close_prices, pd.Series):
        raise ValueError("close_prices must be a pandas Series indexed by datetime")

    rv = daily_rv_from_close(close_prices)
    df = build_har_features(rv)
    if df.shape[0] < 60:
        raise ValueError("Not enough data after rolling windows to fit HAR (need ~60+ rows).")

    model, res_var = fit_har_logrv(df)
    last_row = df.iloc[-1]
    f_RV = forecast_next_daily_rv(model, res_var, last_row)
    vol_annual = annualize_vol_from_daily_rv(f_RV, trading_days=trading_days)
    return vol_annual, f_RV, model


# ---------------- Example usage ----------------
if __name__ == "__main__":

    mt5_conn = MT5Connector()
    underlying_symbol = "PETR4"
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        exit()  

    spot_data = mt5_conn.get_data(underlying_symbol)
    if spot_data is None:
        print("Failed to get historical data")
        exit()
    else:
        print(f"Retrieved {spot_data.head(1)} for {underlying_symbol}") 

    # Example: use daily closes from Yahoo (replace with your series)

    server = mt5_conn.get_account_info().server
    print(f"Connected to MT5 server: {server}")

    prices = spot_data["close"]
    print(f"Using {len(prices)} daily close prices for HAR volatility forecast")
    vol_ann, f_RV_daily, fitted = har_annual_vol_forecast_from_prices(prices)
    print(f"Next-day forecasted daily RV: {f_RV_daily:.6e}")
    print(f"Annualized vol (HAR forecast): {vol_ann:.2%}")
   # print("HAR coefficients:")
   # print(fitted.params)