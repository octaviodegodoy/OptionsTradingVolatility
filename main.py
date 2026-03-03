from datetime import datetime, time
import logging
import asyncio
from constants import ASSET_SYMBOL, DIFF_IV_GARCH_PUTS_THRESHOLD_PCT, GARCH_SAMPLE_SIZE, IV_DIFF_THRESHOLD_CALLS, STEEP_THRESHOLD
from mt5_connector import MT5Connector
from functions.quant_functions import QuantCalculation
import pandas as pd
import numpy as np
from utils import Utils


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    utils = Utils()
    
    # get garch volatility
    quant_calc = QuantCalculation()
   

    # get options chain
    selected_asset = mt5_conn.symbol_select(ASSET_SYMBOL[2], True)
    if not selected_asset:
        logger.error(f"Failed to select {ASSET_SYMBOL[2]}")
        return None
    symbol_info = mt5_conn.get_symbol_info(ASSET_SYMBOL[2])
    chain_options = mt5_conn.get_option_names_by_expiration_time(ASSET_SYMBOL[2])
    logger.info(f"Options Chain for {ASSET_SYMBOL[2]} retrieved {len(chain_options.values())} options.")
    expiration_time = list(chain_options.keys())[0]

    while selected_asset:
       spot_prices_data = mt5_conn.get_data(ASSET_SYMBOL[2], mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0)["close"].values
       garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100
       logger.info(f"GARCH Volatility : {garch_vol:.2f}%")

       logger.info(f"Selected Expiration Time: {datetime.fromtimestamp(expiration_time)}")
       calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)

       if not calls_dict or not puts_dict:
           logger.warning("calls_dict or puts_dict is empty, skipping iteration")
           await asyncio.sleep(15)
           continue
   
       call_iv_dict = {k: v['iv'] for k, v in calls_dict.items()}
       put_iv_dict = {k: v['iv'] for k, v in puts_dict.items()}
   
       print(f"Put IV dict: {put_iv_dict}")
       print(f"Call IV dict: {call_iv_dict}")
   
       real_delta_calls = np.array(list(call_iv_dict.keys()))
       real_delta_puts  = np.array(list(put_iv_dict.keys()))
       real_iv_calls = np.array(list(call_iv_dict.values()))
       real_iv_puts  = np.array(list(put_iv_dict.values()))

       result = quant_calc.are_puts_steeper(real_delta_puts, real_iv_puts, real_delta_calls, real_iv_calls, STEEP_THRESHOLD)
       
       atm_price = (symbol_info.bid + symbol_info.ask) / 2
       print(f"ATM strike price for {ASSET_SYMBOL[2]} is approximately {atm_price}")
   
       put_strikes = [v['strike'] for v in puts_dict.values()]
       put_strikes_and_ivs = [(v['strike'], v['iv']) for v in puts_dict.values()]
       put_strikes = set(put_strikes)
       sorted_strikes = sorted(put_strikes)
       print(f"All put strikes : {sorted(put_strikes)}")
       atm_strike = min(sorted_strikes, key=lambda x: abs(x - atm_price))
   
       print(f"ATM strike determined from available strikes: {atm_strike}")
       atm_strikes = sorted_strikes[max(0, sorted_strikes.index(atm_strike)-1):sorted_strikes.index(atm_strike)+1]
       print(f"ATM strikes: {atm_strikes}")
   
       atm_ivs = [(iv, strike) for strike, iv in put_strikes_and_ivs if strike in atm_strikes]
       print(f"ATM IVs: {atm_ivs}")
   
       min_iv_strike = min(atm_ivs, key=lambda x: x[0])
       put_atm_delta = next((k for k, v in puts_dict.items() if v['strike'] == min_iv_strike[1]), None)
       print(f"Minimum IV at ATM strikes: {min_iv_strike[0]:.2f}% at strike {min_iv_strike[1]} with delta {put_atm_delta:.2f}")
       iv_garch_diff_pct = min_iv_strike[0] - garch_vol
       print(f"Strike with minimum IV: {min_iv_strike[1]}, IV: {min_iv_strike[0]} and GARCH volatility: {garch_vol:.2f}% and puts steeper? {result} and min IV at ATM puts is {iv_garch_diff_pct:.2f}% different from GARCH volatility (threshold was {DIFF_IV_GARCH_PUTS_THRESHOLD_PCT} pp)")
       put_name_min_iv = next((v['option_name'] for v in puts_dict.values() if v['strike'] == min_iv_strike[1]), None)
       ## verify put side steepness and if IV of ATM puts is significantly lower than GARCH volatility
       print(f"Put option with minimum IV at ATM strikes: {put_name_min_iv} and steep threshold is {STEEP_THRESHOLD} pp/delta")
      
       put_condition = iv_garch_diff_pct <= DIFF_IV_GARCH_PUTS_THRESHOLD_PCT or result

       print(f"Put condition for buying: {put_condition} (IV difference from GARCH: {iv_garch_diff_pct:.2f}% and puts steeper? {result})")       

       avg_call_delta = sum(real_delta_calls) / len(real_delta_calls)
       std_call_delta = (sum((x - avg_call_delta) ** 2 for x in real_delta_calls) / len(real_delta_calls)) ** 0.5

       # Find closest deltas to ±1 std from mean
       lower_bound = avg_call_delta - std_call_delta
       upper_bound = avg_call_delta + std_call_delta
       closest_lower = min([d for d in real_delta_calls if d > 0.25], key=lambda x: abs(x - lower_bound), default=None)
       closest_upper = min([d for d in real_delta_calls if d < 0.75], key=lambda x: abs(x - upper_bound), default=None)
     
       # Calculate IV difference
 
       if closest_lower is not None and closest_upper is not None:
            iv_upper = calls_dict[closest_upper]['iv']
            iv_lower = calls_dict[closest_lower]['iv']
            iv_diff = iv_lower - iv_upper 
       else:
            iv_diff = None

       print(f"IV call diff: {iv_diff:.2f} iv upper {iv_upper:.2f} and strike {calls_dict[closest_upper]['strike']} at delta {closest_upper} iv lower {iv_lower:.2f} at delta {closest_lower} and strike {calls_dict[closest_lower]['strike']}" if iv_diff is not None else "N/A (IV difference)")
    
       call_buy = calls_dict[closest_upper]['option_name']
       call_sell = calls_dict[closest_lower]['option_name']
       orders_type = [mt5_conn.ORDER_TYPE_BUY, mt5_conn.ORDER_TYPE_SELL] # or "SELL"

       call_condition = iv_diff is not None and iv_diff <= IV_DIFF_THRESHOLD_CALLS

       min_amount = 100
       put_atm_delta = abs(put_atm_delta) if put_atm_delta is not None else 0.0
       put_amount = float(round(min_amount/put_atm_delta/min_amount)*min_amount if put_atm_delta is not None and put_atm_delta != 0 else min_amount)
       call_delta = closest_upper - closest_lower if closest_upper is not None and closest_lower is not None else 0 
       call_amount = float(round(min_amount/call_delta/min_amount)*min_amount if call_delta != 0 else min_amount) # adjust call amount based on delta difference to maintain a more balanced position
           
       puts_positions_total = utils.put_options_count()
       print(f"Current open put positions: {puts_positions_total}")
       call_positions_total = utils.call_options_count()
       print(f"Current open call positions: {call_positions_total}")

       put_order_allowed = puts_positions_total == 0 and put_condition
       call_order_allowed = call_positions_total < 2 and call_condition

       print(f"Checking order for put option {put_name_min_iv} with amount {put_amount} based on ATM put delta {put_atm_delta:.2f} steep {result} and IV difference from GARCH {iv_garch_diff_pct:.2f}%") 

       if put_order_allowed:          
          logger.info(f"Puts are steeper than calls with a slope difference of at least {STEEP_THRESHOLD} pp/delta.")
          symbol_info = mt5_conn.get_symbol_info(ASSET_SYMBOL[2])
          mt5_conn.place_order(put_name_min_iv,MT5Connector.ORDER_TYPE_BUY, put_amount, symbol_info.ask, 10, str(min_iv_strike[0]))
       else:
          logger.info(f"Condition not met for put condition: {put_condition} (IV difference from GARCH: {iv_garch_diff_pct:.2f}% and puts steeper? {result})")

       print(f"Calculated call amount based on delta difference: {call_amount} (delta difference: {call_delta:.2f}) for call option name {call_buy} at delta {closest_upper} and call option name {call_sell} at delta {closest_lower} with IV difference between strikes of {iv_diff:.2f}%")
       if call_order_allowed:
            print(f"Placing orders for call spread: Buy {call_buy} and Sell {call_sell} and IV difference is {iv_diff:.2f}% which is less than or equal to {IV_DIFF_THRESHOLD_CALLS}%")
            mt5_conn.place_order_vertical(call_buy, call_sell, orders_type, call_amount, iv_upper, iv_lower)
       else:
            logger.info(f"Condition not met for call condition: {call_condition} diff threshold is {IV_DIFF_THRESHOLD_CALLS} (IV difference between call strikes: {iv_diff:.2f}%)")
            

       """    
       print(f"Average Call Delta: {avg_call_delta:.2f} ± {std_call_delta:.2f} delta diff {call_delta_diff:.2f}")
       print(f"Closest existing deltas: Lower={closest_lower:.2f} option name {real_delta_calls[closest_lower]['option_name']} with (IV={real_delta_calls[closest_lower]['iv']:.2f}%), Upper={closest_upper:.2f} and option name {real_delta_calls[closest_upper]['option_name']}  with (IV={real_delta_calls[closest_upper]['iv']:.2f}%) diff IV={iv_diff:.2f}%" if iv_diff is not None else "N/A  (IV difference)")
       """ 
       #F = utils.get_factor_from_expiration_time(expiration_time)
       #logger.info(f"Tenor Factor from utils function: {F}")
       #T = utils.get_tenor(expiration_time)
       #logger.info(f"Tenor is: {T} weekdays to expiry. and expiration time is {datetime.fromtimestamp(expiration_time)}")
       #r = black_scholes_calculator.get_interpolated_rate(pd.to_datetime(expiration_time, unit='s'))
       #logger.info(f"Interest Rate for {datetime.fromtimestamp(expiration_time).date()}: {r}")
       await asyncio.sleep(25)    
    

asyncio.run(main())