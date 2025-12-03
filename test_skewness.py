import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

call_list = [18.12, 14.39, 22.46, 21.50, 21.82, 19.56, 20.62, 21.46, 20.65]
put_list  = [20.15, 24.09, 25.05, 23.76, 24.32, 25.09, 18.24, 30.30]

# If your data correspond 1:1 by index, use overlapping length:
n = min(len(call_list), len(put_list))
calls = np.array(call_list[:n])
puts  = np.array(put_list[:n])

def summary(arr):
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std_sample": float(np.std(arr, ddof=1)),
        "skew_sample": float(stats.skew(arr, bias=False))
    }

summary_calls = summary(calls)
summary_puts  = summary(puts)

spreads = puts - calls   # elementwise put - call
spread_summary = {
    "n": len(spreads),
    "mean_spread": float(np.mean(spreads)),
    "median_spread": float(np.median(spreads)),
    "min_spread": float(np.min(spreads)),
    "max_spread": float(np.max(spreads)),
    "std_spread": float(np.std(spreads, ddof=1))
}

# Paired t-test (assumes approx normal) and Wilcoxon signed-rank (non-parametric)
tt = stats.ttest_rel(puts, calls)
wil = stats.wilcoxon(puts, calls)

print("Calls summary:", summary_calls)
print("Puts summary:", summary_puts)
print("Spreads (put - call) summary:", spread_summary)
print("Paired t-test:", tt)
print("Wilcoxon signed-rank test:", wil)

# Optional plots:
plt.figure(figsize=(8,4))
x = np.arange(n)  # index or strikes/deltas if you have them
plt.plot(x, calls, marker='o', label='Calls IV (%)')
plt.plot(x, puts,  marker='o', label='Puts IV (%)')
plt.legend()
plt.title('IV curves (index order)')
plt.xlabel('Index (strike/delta order)')
plt.ylabel('Implied vol (%)')
plt.grid(True)
plt.show()

plt.figure(figsize=(6,3))
plt.bar(x, spreads)
plt.axhline(0, color='k')
plt.title('Put - Call IV spread (%) by index')
plt.xlabel('Index')
plt.ylabel('Put - Call (pp)')
plt.grid(True)
plt.show()