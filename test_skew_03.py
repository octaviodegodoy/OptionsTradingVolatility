"""
Reproduce the linear-fit skew comparison between puts and calls.

Assumptions made (as used in the previous result):
- We drop the last put IV (30.30) because there are 7 put deltas but 8 put IVs.
- Put deltas are treated by absolute value.
- For a fair comparison we fit over comparable delta ranges:
    * use all puts (abs deltas 0.18..0.60)
    * use only call points whose delta is inside that same range (0.18..0.60)
- Linear fit model: IV = intercept + slope * delta
    * slope is reported in percentage points (pp) of IV per unit delta.
    * a negative slope means IV decreases as delta increases.

Run:
    python skew_analysis.py
"""
import numpy as np

from constants import ASSET_SYMBOL, GARCH_SAMPLE_SIZE
from mt5_connector import MT5Connector
from functions.quant_functions import QuantCalculation
from utils import Utils

def are_puts_steeper(delta_puts, put_ivs, delta_calls, call_ivs, min_diff_pp_per_delta=0.0):
    """
    Check if put IVs have a steeper slope than call IVs when fitted linearly.
    
    Args:
        delta_puts: array of put deltas (negative values)
        put_ivs: array of put implied volatilities (%)
        delta_calls: array of call deltas (positive values)
        call_ivs: array of call implied volatilities (%)
        min_diff_pp_per_delta: minimum difference in absolute slope (pp/delta) 
                               required for puts to be considered steeper.
                               Default is 0.0 (any difference counts).
    
    Returns:
        bool: True if abs(put_slope) - abs(call_slope) >= min_diff_pp_per_delta,
              False otherwise
    
    Example:
        # Require puts to be at least 1.0 pp/delta steeper than calls
        result = are_puts_steeper(delta_puts, put_ivs, delta_calls, call_ivs, min_diff_pp_per_delta=1.0)
    
    Note:
        - Put deltas are converted to absolute values for comparison
        - Both fits are done over the same delta range (the put delta range)
        - Steeper means larger absolute slope value (more negative)
    """
    # Handle data length mismatch (drop extra IVs if needed)


    put_ivs = put_ivs[:len(delta_puts)]
    
    # Convert put deltas to absolute values
    puts_delta = np.abs(delta_puts)
    
    # Determine delta range from puts
    delta_min = puts_delta.min()
    delta_max = puts_delta.max()
    
    # Filter calls to same delta range
    calls_mask = (delta_calls >= delta_min) & (delta_calls <= delta_max)
    calls_delta_filtered = delta_calls[calls_mask]
    calls_iv_filtered = call_ivs[calls_mask]
    
    # Linear fit: IV = intercept + slope * delta
    slope_put, _ = np.polyfit(puts_delta, put_ivs, 1)
    slope_call, _ = np.polyfit(calls_delta_filtered, calls_iv_filtered, 1)
    
    # Calculate difference in absolute slopes
    slope_diff = abs(slope_put) - abs(slope_call)
    
    # Puts are steeper if the difference meets the minimum threshold
    return slope_diff >= min_diff_pp_per_delta


mt5_conn = MT5Connector()
utils = Utils()
quant_calc = QuantCalculation()

asset_symbol = ASSET_SYMBOL[0]
spot_prices_data = mt5_conn.get_data(ASSET_SYMBOL[0], mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0)["close"].values
garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100


chain_options = mt5_conn.get_option_names_by_expiration_time(asset_symbol)
print(f"Retrieved expiration chain for {asset_symbol} with {chain_options} options.")
symbol_info = mt5_conn.get_symbol_info(asset_symbol)
atm_price = (symbol_info.bid + symbol_info.ask) / 2
print(f"ATM strike price for {asset_symbol} is approximately {atm_price}")
print("Options Chain for", asset_symbol, "retrieved", chain_options.values(), "options.")
call_deltas_dict, put_deltas_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
print("---- Retrieved Options Data ----")
print(f"Puts dict first 5: {list(put_deltas_dict.items())[:5]}")
call_strikes = [v['strike'] for v in call_deltas_dict.values()]
put_strikes = [v['strike'] for v in put_deltas_dict.values()]
all_strikes = set(call_strikes) | set(put_strikes)
sorted_strikes = sorted(all_strikes)
print(f"All strikes (merged): {sorted(all_strikes)}")
atm_strike = min(sorted_strikes, key=lambda x: abs(x - atm_price))

print(f"ATM strike determined from available strikes: {atm_strike}")
atm_strikes = sorted_strikes[max(0, sorted_strikes.index(atm_strike)-1):sorted_strikes.index(atm_strike)+1]
print(f"ATM strikes: {atm_strikes}")
ivs_in_atm_strikes_puts = [v['iv'] for v in put_deltas_dict.values() if v['strike'] in atm_strikes]
print(f"IV at ATM strike for puts: {ivs_in_atm_strikes_puts}")
print(f"GARCH Volatility : {garch_vol:.2f}%")
# Get the delta (key) from call_deltas_dict where strike is atm_strike
atm_put_delta = next((delta for delta, v in put_deltas_dict.items() if v['strike'] == atm_strike), None)

if atm_put_delta is not None:
    print(f"\nATM Put Delta: {atm_put_delta:.3f} at strike {atm_strike}")
    print(f"ATM Put IV: {put_deltas_dict[atm_put_delta]['iv']:.2f}% and put strike: {put_deltas_dict[atm_put_delta]['strike']}")
else:
    print(f"\nNo put option found at ATM strike {atm_strike}")

print(f"Options Data Retrieved: calls {call_deltas_dict.keys()}, puts {put_deltas_dict.keys()}")
call_iv_dict = {k: v['iv'] for k, v in call_deltas_dict.items()}
put_iv_dict = {k: v['iv'] for k, v in put_deltas_dict.items()}
 

real_delta_calls = np.array(list(call_iv_dict.keys()))
real_delta_puts  = np.array(list(put_iv_dict.keys()))
real_iv_calls = np.array(list(call_iv_dict.values()))
real_iv_puts  = np.array(list(put_iv_dict.values()))


# Original data
delta_puts = real_delta_puts # np.array([-0.18, -0.25, -0.31, -0.39, -0.47, -0.56, -0.6])
put_list_all = real_iv_puts # np.array([20.15, 24.09, 25.05, 23.76, 24.32, 25.09, 18.24, 30.30])  # 8 IVs, will drop last

delta_calls = real_delta_calls
call_list = real_iv_calls

# Prepare puts: drop the extra IV (assumption)
puts_iv = put_list_all[:len(delta_puts)]  # take first 7 IVs
puts_delta = np.abs(delta_puts)           # use absolute delta for puts

# Determine delta range covered by puts
delta_min = puts_delta.min()
delta_max = puts_delta.max()

# Filter calls to the same delta range to compare slopes on comparable region
calls_mask = (delta_calls >= delta_min) & (delta_calls <= delta_max)
calls_delta_sel = delta_calls[calls_mask]
calls_iv_sel = call_list[calls_mask]

# Sort data by delta for nicer output / stability (not required for fitting)
sort_idx_puts = np.argsort(puts_delta)
puts_delta_sorted = puts_delta[sort_idx_puts]
puts_iv_sorted = puts_iv[sort_idx_puts]

sort_idx_calls = np.argsort(calls_delta_sel)
calls_delta_sorted = calls_delta_sel[sort_idx_calls]
calls_iv_sorted = calls_iv_sel[sort_idx_calls]

# Linear fit using numpy.polyfit (degree 1): returns [slope, intercept] for polyfit(x, y, 1)
slope_put, intercept_put = np.polyfit(puts_delta_sorted, puts_iv_sorted, 1)
slope_call, intercept_call = np.polyfit(calls_delta_sorted, calls_iv_sorted, 1)

# Convert slope to "per 0.1 delta" for easier intuition
slope_put_per_0p1 = slope_put * 0.1
slope_call_per_0p1 = slope_call * 0.1

# Calculate the absolute difference
slope_diff = abs(slope_put) - abs(slope_call)

# Print results
print("Data used for puts (delta, IV):")
for d, iv in zip(puts_delta_sorted, puts_iv_sorted):
    print(f"  {d:.2f} , {iv:.2f}%")

print("\nData used for calls (delta, IV) filtered to put range:")
for d, iv in zip(calls_delta_sorted, calls_iv_sorted):
    print(f"  {d:.2f} , {iv:.2f}%")

print("\nLinear fit results (IV = intercept + slope * delta):")
print(f" Puts: slope = {slope_put:.3f} pp per unit delta, intercept = {intercept_put:.3f}")
print(f" Calls: slope = {slope_call:.3f} pp per unit delta, intercept = {intercept_call:.3f}")

print("\nInterpreting slope per 0.1 change in delta (more intuitive):")
print(f" Puts change ≈ {slope_put_per_0p1:.3f} pp per 0.1 delta")
print(f" Calls change ≈ {slope_call_per_0p1:.3f} pp per 0.1 delta")

print("\nDifference in absolute slopes:")
print(f" |slope_put| - |slope_call| = {slope_diff:.3f} pp per unit delta")
print(f" Difference per 0.1 delta = {slope_diff * 0.1:.3f} pp per 0.1 delta")

# Test the function with different thresholds
print("\n" + "="*60)
print("FUNCTION TESTS:")
print("="*60)

test_thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
for threshold in test_thresholds:
    result = are_puts_steeper(delta_puts, put_list_all, delta_calls, call_list, 
                             min_diff_pp_per_delta=threshold)
    print(f"Min threshold = {threshold:.2f} pp/delta: Puts steeper? {result}")
# Example of predicted IV at a given delta (optional demonstration)
target_delta = 0.25
pred_put_iv_at_025 = intercept_put + slope_put * target_delta
pred_call_iv_at_025 = intercept_call + slope_call * target_delta
print(f"\nPredicted IV at delta={target_delta:.2f}: put={pred_put_iv_at_025:.2f}%, call={pred_call_iv_at_025:.2f}%")
print(f"Predicted put - call at {target_delta}: {pred_put_iv_at_025 - pred_call_iv_at_025:.2f}%")


# --- End of script ---

# Expected (approx) outputs based on the assumptions used earlier:
# (These are the numbers you should see when running this script)
#
# Puts: slope ≈ -1.713 pp per unit delta
# Calls: slope ≈ -0.200 pp per unit delta
# Difference (|put| - |call|) ≈ 1.513 pp per unit delta
# Per 0.1 delta:
#   Puts ≈ -0.171 pp per 0.1 delta
#   Calls ≈ -0.020 pp per 0.1 delta
#
# So puts are substantially steeper (in magnitude) than calls on the compared delta range.