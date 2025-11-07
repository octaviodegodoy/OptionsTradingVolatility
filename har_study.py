#!/usr/bin/env python3
"""
HAR model example with random (synthetic) realized volatility data.

- Simulates a persistent daily realized variance (RV) process via AR(1) on log(RV).
- Builds HAR features: daily (d), weekly (w), monthly (m) using averages of levels, then logs.
- Fits OLS: log(RV_{t+1}) ~ const + log(RV_t) + log(avg_5 RV) + log(avg_22 RV)
- Produces one-step-ahead test forecasts with bias-corrected back-transform to levels.
- Reports coefficients and common forecast metrics (RMSE on log, QLIKE on levels).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Optional plotting; set to True to see a plot
PLOT = True

def simulate_log_rv(T=1000, mu=-1.5, phi=0.97, sigma_eta=0.25, seed=42):
    """
    Simulate daily log realized variance log(RV_t) via AR(1).
    RV_t = exp(logRV_t)
    """
    rng = np.random.default_rng(seed)
    logRV = np.empty(T)
    logRV[0] = mu / (1 - phi)  # start at unconditional mean
    for t in range(1, T):
        logRV[t] = mu + phi * logRV[t - 1] + rng.normal(0.0, sigma_eta)
    RV = np.exp(logRV)  # realized variance (not volatility)
    return logRV, RV

def build_har_frame(RV):
    """
    Build a DataFrame with:
    - y: log(RV_t)
    - d: log(RV_t)
    - w: log( (1/5) * sum_{i=0..4} RV_{t-i} )
    - m: log( (1/22) * sum_{i=0..21} RV_{t-i} )
    - y_fwd: log(RV_{t+1})
    """
    rv = pd.Series(RV).rename("RV").reset_index(drop=True)
    y = np.log(rv)
    w = np.log(rv.rolling(5).mean())
    m = np.log(rv.rolling(22).mean())

    df = pd.DataFrame(
        {
            "y": y,              # log(RV_t)
            "d": y,              # daily term at t
            "w": w,              # weekly (avg of levels) then log
            "m": m,              # monthly (avg of levels) then log
        }
    )
    df["y_fwd"] = df["y"].shift(-1)  # target: log(RV_{t+1})
    df = df.dropna().reset_index(drop=True)
    return df

def fit_har(df, test_horizon=200):
    """
    Fit HAR on training set and evaluate on test set.
    Returns fitted model, metrics, forecasts, and aligned test targets.
    """
    # Train/test split preserving order
    n = len(df)
    test_horizon = min(test_horizon, n // 3)  # keep enough train data
    split = n - test_horizon

    X = sm.add_constant(df[["d", "w", "m"]])
    y = df["y_fwd"]

    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = sm.OLS(y_train, X_train).fit()

    # In-sample residual variance (for bias-corrected back-transform)
    yhat_train = model.predict(X_train)
    res_var = float(np.mean((yhat_train - y_train) ** 2))

    # One-step-ahead predictions on test
    yhat_test_log = model.predict(X_test)

    # Back-transform to RV levels with bias correction
    f_RV_test = np.exp(yhat_test_log + 0.5 * res_var)
    RV_test_actual = np.exp(y_test)

    # Metrics
    rmse_log = float(np.sqrt(np.mean((yhat_test_log - y_test) ** 2)))
    qlike = float(np.mean(RV_test_actual / f_RV_test - np.log(RV_test_actual / f_RV_test) - 1.0))

    metrics = {
        "rmse_log": rmse_log,
        "qlike": qlike,
        "train_size": int(split),
        "test_size": int(len(y_test)),
        "res_var_train": res_var,
    }

    results = {
        "model": model,
        "metrics": metrics,
        "y_test_log": y_test.reset_index(drop=True),
        "yhat_test_log": yhat_test_log.reset_index(drop=True),
        "RV_test_actual": RV_test_actual.reset_index(drop=True),
        "RV_test_forecast": pd.Series(f_RV_test).reset_index(drop=True),
    }
    return results

def main():
    # 1) Simulate a persistent realized variance process
    logRV, RV = simulate_log_rv(
        T=1000,     # number of days
        mu=-1.5,    # long-run mean of log(RV)
        phi=0.97,   # persistence
        sigma_eta=0.25,
        seed=42
    )

    # 2) Build HAR features and target
    df = build_har_frame(RV)

    # 3) Fit and evaluate
    results = fit_har(df, test_horizon=200)
    model = results["model"]
    metrics = results["metrics"]

    print("\n=== HAR coefficients (log RV model) ===")
    print(model.params.rename({"const": "beta0", "d": "beta_d", "w": "beta_w", "m": "beta_m"}))
    print("\n=== In-sample summary ===")
    print(model.summary().tables[0])
    print(model.summary().tables[1])

    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # 4) Show a few example forecasts in levels (RV)
    out = pd.DataFrame(
        {
            "RV_actual": results["RV_test_actual"],
            "RV_forecast": results["RV_test_forecast"],
        }
    )
    print("\n=== Sample of test set forecasts (levels: realized variance) ===")
    print(out.head(10).to_string(index=False))

    # Optional: plot
    if PLOT:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(out["RV_actual"].values, label="Actual RV (test)")
            plt.plot(out["RV_forecast"].values, label="Forecast RV (HAR)", alpha=0.8)
            plt.legend()
            plt.title("HAR one-step-ahead forecasts on synthetic realized variance")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Plot skipped: {e})")

if __name__ == "__main__":
    main()