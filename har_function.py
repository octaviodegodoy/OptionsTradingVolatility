#!/usr/bin/env python3
"""
Minimal HAR example: compute and print only the last one-step-ahead RV forecast.

- Simulates daily realized variance (RV) from an AR(1) on log(RV)
- Builds HAR predictors: log(daily RV), log(weekly avg RV), log(monthly avg RV)
- Fits OLS via numpy (no statsmodels) on all but the final row with target log(RV_{t+1})
- Bias-corrects the back-transform to RV levels and prints the last forecast
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_log_rv(T: int = 300, mu: float = -1.5, phi: float = 0.97, sigma_eta: float = 0.25, seed: int = 42) -> np.ndarray:
    """
    Simulate daily log(RV) via AR(1), then return RV levels (variance).
    """
    rng = np.random.default_rng(seed)
    log_rv = np.empty(T)
    log_rv[0] = mu / (1.0 - phi)
    for t in range(1, T):
        log_rv[t] = mu + phi * log_rv[t - 1] + rng.normal(0.0, sigma_eta)
    rv = np.exp(log_rv)  # realized variance
    return rv


def har_last_forecast_from_rv(rv: np.ndarray) -> float:
    """
    Given a series of realized variance (RV) levels, fit a HAR model on log(RV)
    and return the last one-step-ahead RV forecast (level, not log).

    Returns:
        float: The bias-corrected forecast of RV_{T+1} based on the last available predictors.
    """
    s = pd.Series(np.asarray(rv, dtype=float)).dropna().reset_index(drop=True)

    # Need at least 22 days for monthly average + 1 step ahead
    if len(s) < 30:
        raise ValueError("Provide at least ~30 observations to compute weekly/monthly averages robustly.")

    # HAR predictors: average levels over 5 and 22 days, then log
    d = np.log(s)
    w = np.log(s.rolling(5).mean())
    m = np.log(s.rolling(22).mean())

    # Target is next day's log(RV)
    y_fwd = d.shift(-1)

    df = pd.DataFrame({"const": 1.0, "d": d, "w": w, "m": m, "y_fwd": y_fwd}).dropna()

    # Use all but the last row to fit; last row to forecast
    if len(df) < 2:
        raise ValueError("Not enough rows after rolling windows to fit and forecast.")

    X = df[["const", "d", "w", "m"]].to_numpy()
    y = df["y_fwd"].to_numpy()

    X_train, y_train = X[:-1], y[:-1]
    X_last = X[-1]

    # OLS via least squares
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    # In-sample residual variance for bias-corrected back-transform
    yhat_train = X_train @ beta
    res_var = float(np.mean((y_train - yhat_train) ** 2))

    # Last one-step-ahead forecast (in log space)
    yhat_last_log = float(X_last @ beta)

    # Bias-corrected forecast in RV levels
    last_forecast_rv = float(np.exp(yhat_last_log + 0.5 * res_var))
    return last_forecast_rv


def main():
    # Example usage with synthetic data
    rv = simulate_log_rv(T=300, mu=-1.5, phi=0.97, sigma_eta=0.25, seed=42)
    last_forecast = har_last_forecast_from_rv(rv)
    # Print only the last RV forecast
    print(f"{last_forecast:.8f}")


if __name__ == "__main__":
    main()