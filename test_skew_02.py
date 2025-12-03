import numpy as np

call_list = [18.12,14.39,22.46,21.50,21.82,19.56,20.62,21.46,20.65]
put_list  = [20.15,24.09,25.05,23.76,24.32,25.09,18.24,30.30]

# align length (use first 8 of calls to match 8 puts)
n = min(len(call_list), len(put_list))
calls = np.array(call_list[:n])
puts  = np.array(put_list[:n])

x = np.arange(n)        # index proxy for strike/delta
xbar = x.mean()

# slope = sum((x-xbar)*(y-ybar)) / sum((x-xbar)**2)
def slope(x, y):
    xbar = x.mean()
    ybar = y.mean()
    num = ((x - xbar) * (y - ybar)).sum()
    den = ((x - xbar)**2).sum()
    return num / den

slope_calls = slope(x, calls)
slope_puts  = slope(x, puts)

print("slope_calls:", slope_calls)
print("slope_puts:", slope_puts)
print("put - call slope diff:", slope_puts - slope_calls)
print("relative diff (puts vs calls):", (slope_puts / slope_calls - 1) * 100, "%")