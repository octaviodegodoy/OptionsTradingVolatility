"""
Example usage for c_spline with date -> percentage sample data.

Assumes c_spline(xi, yi, x_query) is importable from c_spline.py
"""

from cubic_spline import c_spline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- Input: your date / percent pairs ---
pairs = [
    ("11/19/2025",  "8.65%"),
    ("12/1/2025",  "17.25%"),
    ("1/2/2026",  "10.00%"),
    ("2/1/2026", "12.50%"),
]

# Parse into pandas objects for convenience
dates = pd.to_datetime([p[0] for p in pairs], format="%m/%d/%Y")
percents = pd.Series([p[1] for p in pairs], index=dates)

# Convert percent strings to numeric fractions (0.1492)
yi_frac = percents.str.rstrip("%").astype(float) / 100.0

# Convert dates to numeric ordinals for spline (float)
xi_ord = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)

# For clarity, also keep last_close-style values (percent as number)
yi_percent_number = percents.str.rstrip("%").astype(float).to_numpy(dtype=float)

print("xi (ordinals):", xi_ord)
print("yi (fractions):", yi_frac.to_numpy())

# --- 1) Evaluate spline at a scalar date (e.g., 2025-08-15) ---
query_date = pd.to_datetime("1/16/2026")
query_ord = float(query_date.toordinal())
y_at_query = c_spline(xi_ord, yi_frac.to_numpy(), query_ord)
print(f"Spline at {query_date.date()} -> {y_at_query:.16f} (fraction) -> {y_at_query*100:.14f}%")

# --- 2) Evaluate spline on a vector of dates for plotting ---
xq_ordinals = np.linspace(xi_ord.min(), xi_ord.max(), 200)
yq = c_spline(xi_ord, yi_frac.to_numpy(), xq_ordinals)

# Convert ordinals back to datetimes for plotting ticks
xq_dates = [pd.Timestamp.fromordinal(int(o)) for o in xq_ordinals]

# Plot: percent (converted back to percent for axis readability)
plt.figure(figsize=(8, 4))
plt.plot(dates, yi_percent_number, "o", label="input data (percent)")
plt.plot(xq_dates, np.array(yq) * 100.0, "-", label="c_spline interpolation")
plt.xlabel("Date")
plt.ylabel("Percent (%)")
plt.title("Natural cubic spline interpolation of rates")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()

# --- 3) Provide a pandas Series of interpolated daily values (optional) ---
# Example: create daily range between min and max and evaluate
daily_index = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
daily_ord = daily_index.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
daily_y = c_spline(xi_ord, yi_frac.to_numpy(), daily_ord)
daily_series = pd.Series(daily_y * 100.0, index=daily_index)  # percent numbers

print("\nSample of interpolated daily series:")
print(daily_series.head(7))

# Optionally save to CSV
# daily_series.to_csv("interpolated_rates_daily.csv", header=["percent"])