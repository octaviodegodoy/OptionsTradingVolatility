import numpy as np
from utils import Utils
utils = Utils()
from mt5_connector import MT5Connector
from constants import ASSET_SYMBOL

mt5_conn = MT5Connector()

for asset in ASSET_SYMBOL:
    selected=mt5_conn.symbol_select(asset,True) 
    if not selected: 
        print(f"Failed to select {asset}")
   # chain_options = mt5_conn.get_option_names_by_expiration_time(asset)
   # symbol_info = mt5_conn.get_symbol_info(asset)
   # print("Options Chain for", asset, "retrieved", chain_options.values(), "options.")
    
asset_symbol = ASSET_SYMBOL[2]
chain_options = mt5_conn.get_option_names_by_expiration_time(asset_symbol)
symbol_info = mt5_conn.get_symbol_info(asset_symbol)
print("Options Chain for", asset_symbol, "retrieved", chain_options.values(), "options.")
call_deltas_dict, put_deltas_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
call_iv_dict = {k: v['iv'] for k, v in call_deltas_dict.items()}
put_iv_dict = {k: v['iv'] for k, v in put_deltas_dict.items()}
print(f"Options Data Retrieved: calls {call_iv_dict}, puts {put_iv_dict}")

n_options = min(len(call_iv_dict), len(put_iv_dict))
call_deltas = np.array(list(call_iv_dict.keys())[:n_options])
put_deltas  = np.array(list(put_iv_dict.keys())[:n_options])

#print("Call Deltas:", call_deltas)
#print("Put Deltas:", put_deltas)

x = np.arange(n_options)  # index proxy for strike/delta
xbar = x.mean()

def slope(x, y):
    xbar = x.mean()
    ybar = y.mean()
    num = ((x - xbar) * (y - ybar)).sum()
    den = ((x - xbar)**2).sum()
    return num / den

slope_calls = slope(x, call_deltas)
slope_puts  = slope(x, put_deltas)

print("slope_calls:", slope_calls)
print("slope_puts:", slope_puts)
print("put - call slope diff:", slope_puts - slope_calls)
print("relative diff (puts vs calls):", (slope_puts / slope_calls - 1) * 100, "%")