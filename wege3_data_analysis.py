# Ready-to-run GARCH(1,1) analysis for WEGE3 closing prices
# Requires: pandas, numpy, matplotlib, seaborn, arch, statsmodels
# Install: pip install pandas numpy matplotlib seaborn arch statsmodels
# Usage: python wege3_garch_analysis.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

CSV_PATH = "price_data.csv"
DATE_COL = "Date"
CLOSE_COL = "Close"
TRADING_DAYS_PER_YEAR = 252
FORECAST_HORIZON_DAYS = 5

def main():
    df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df[CLOSE_COL] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    df = df.dropna(subset=[CLOSE_COL])

    # compute log returns
    df["logp"] = np.log(df[CLOSE_COL])
    df["ret"] = df["logp"].diff()
    rets = df["ret"].dropna()
    print(f"Observations (daily returns): {len(rets)}")
    print("Sample mean (daily):", rets.mean())
    print("Sample std (daily):", rets.std())

    # ADF test
    adf_res = adfuller(rets)
    print("ADF test statistic:", adf_res[0], "p-value:", adf_res[1])

    # ARCH-LM test
    arch_test = het_arch(rets.dropna())
    print("ARCH LM test (stat, p-value):", arch_test[0], arch_test[1])

    # Fit GARCH(1,1) with Student-t on percent returns
    rets_pct = rets * 100.0
    am = arch_model(rets_pct, mean="Constant", vol="Garch", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    print(res.summary())

    # conditional volatility and annualized
    cond_vol = res.conditional_volatility  # in percent
    daily_vol_decimal = cond_vol / 100.0
    annual_vol_pct = daily_vol_decimal * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0

    df2 = df.loc[rets.index].copy()
    df2["ret"] = rets.values
    df2["cond_vol_daily_pct"] = cond_vol
    df2["cond_vol_annual_pct"] = annual_vol_pct

    # Forecast horizon
    fcast = res.forecast(horizon=FORECAST_HORIZON_DAYS, reindex=False)
    variance_forecast = fcast.variance.values[-1]  # array length = horizon
    daily_variance_forecast = variance_forecast / (100.0**2)
    daily_vol_forecast = np.sqrt(daily_variance_forecast)
    annual_vol_forecast = daily_vol_forecast * np.sqrt(TRADING_DAYS_PER_YEAR)

    print("\nForecast (annualized vol) for next days:")
    for i, v in enumerate(annual_vol_forecast, start=1):
        print(f" Day {i}: {v*100:.2f}%")

    # Diagnostics
    std_resid = res.std_resid
    lb_sq = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
    print("\nLjung-Box on squared standardized residuals (lag=10):")
    print(lb_sq)

    # Save outputs
    out = {
        "n_obs": int(len(rets)),
        "mean_daily_ret": float(rets.mean()),
        "std_daily_ret": float(rets.std()),
        "adf_stat": float(adf_res[0]),
        "adf_pvalue": float(adf_res[1]),
        "arch_lm_stat": float(arch_test[0]),
        "arch_lm_pvalue": float(arch_test[1]),
        "garch_summary": str(res.summary()),
        "last_cond_vol_daily_pct": float(cond_vol.values[-1]),
        "last_cond_vol_annual_pct": float(annual_vol_pct.values[-1]),
        "annual_vol_forecast_pct": [float(v*100) for v in annual_vol_forecast],
        "ljungbox_sq_pvalue": float(lb_sq["lb_pvalue"].iloc[0])
    }
    with open("wege3_garch_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results to wege3_garch_results.json")

    # Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df2[DATE_COL], df2[CLOSE_COL])
    plt.title("WEGE3 Close Price")
    plt.subplot(2, 1, 2)
    plt.plot(df2[DATE_COL], df2["cond_vol_annual_pct"], color="C1")
    plt.title("GARCH(1,1) Conditional Annualized Vol (%)")
    plt.tight_layout()
    plt.savefig("wege3_price_and_vol.png")
    print("Saved plot to wege3_price_and_vol.png")

if __name__ == "__main__":
    main()