
from ast import If
from math import log
from datetime import datetime
import time
from scipy.stats import norm
from scipy.optimize import newton, brentq
from constants import CALL_OPTION, MIN_DAYS_TO_EXPIRY, PUT_OPTION, STRIKE_PRICE_OFFSET, UNIX_DAYS_IN_SECONDS
from functions.quant_functions import QuantCalculation
from mt5_connector import MT5Connector
from functions.original_spline import c_spline
import pandas as pd

from utils import count_weekdays

class BlackScholesCalculator:
    def __init__(self):
        pass

    def d_1(self, F, K, T, sigma):
        return (log(F / K) + sigma ** 2 * (T / 252) / 2) / (sigma * (T / 252) ** (1 / 2))
    
    def d_2(self, F, K, T, sigma):
        return self.d_1(F, K, T, sigma) - sigma * (T / 252) ** (1 / 2)
    

    def fx_delta_call(self, F, K, T, sigma, r, spot):
        x = self.d_1(F, K, T, sigma)
        if spot == 0:
            delta_call = norm.cdf(x)
        else:
            delta_call = F / (spot * r) * norm.cdf(x)
        
        return delta_call
    
    def fx_delta_put(self, F, K, T, sigma, r, spot):
        x = self.d_1(F, K, T, sigma)
        if spot == 0:
            delta_put = norm.cdf(x) - 1
        else:
            delta_put = F / (spot * r) * (norm.cdf(x) - 1)
        
        return delta_put
    
    def fx_delta(self, F, K, T, sigma, r, ID, spot):
        if ID == CALL_OPTION:
            return self.fx_delta_call(F, K, T, sigma, r, spot)
        elif ID == PUT_OPTION:
            return self.fx_delta_put(F, K, T, sigma, r, spot)
    
    def fx_put(self, F, K, T, sigma, r):
        d1_value = self.d_1(F, K, T, sigma)
        d2_value = self.d_2(F, K, T, sigma)

        N_neg_d1 = norm.cdf(-d1_value)
        N_neg_d2 = norm.cdf(-d2_value)
        put_price = (K * N_neg_d2 - F * N_neg_d1) * r  # Assuming a risk-free rate of 5%
        
        if put_price < 0:
            put_price = 0.0

        return put_price
    
    def fx_call(self, F, K, T, sigma, r):
        d1_value = self.d_1(F, K, T, sigma)
        d2_value = self.d_2(F, K, T, sigma)
        N_d1 = norm.cdf(d1_value)
        N_d2 = norm.cdf(d2_value)
        fx_call = (F * N_d1 - K * N_d2) * r

        if fx_call < 0:
            fx_call = 0.0

        return fx_call
    
    def fx_call_vol(self, F, K, T, price, r):
        
        high = 5
        low = 0
        while (high - low) > 0.00000001:
            if self.fx_call(F, K, T, (high + low) / 2, r) > price:
                high = (high + low) / 2
            else:
                low = (high + low) / 2
        
        
        return (high + low) / 2
    
    def fx_put_vol(self, F, K, T, price, r):
        
        high = 5
        low = 0
        while (high - low) > 0.00000001:
            if self.fx_put(F, K, T, (high + low) / 2, r) > price:
                high = (high + low) / 2
            else:
                low = (high + low) / 2
        
        return (high + low) / 2
    
    def fx_vol(self, F, K, T, price, r, ID):
        if ID == CALL_OPTION:
            return self.fx_call_vol(F, K, T, price, r)
        elif ID == PUT_OPTION:
            return self.fx_put_vol(F, K, T, price, r)
        
        # Helper function to get interpolated rate for any future date
    def get_interpolated_rate(self, target_date):
        """Get interpolated interest rate for a specific date"""
         # Get interest rate
        data = mt5_conn.get_data("DI1@", mt5_conn.get_mt5_connector().TIMEFRAME_MN1, 7, 0)
        interest_rates = data.sort_index(ascending=True)

        # Convert timestamps to numeric values (days since first date) for spline interpolation
        time_numeric = (interest_rates['time'] - interest_rates['time'].iloc[0]).dt.total_seconds() / 86400

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        target_numeric = (target_date - interest_rates['time'].iloc[0]).total_seconds() / 86400
        
        rate = c_spline(
            x_data=time_numeric.values,
            y_data=interest_rates['close'].values,
            x_eval=target_numeric
        )
        return rate

    def implied_volatility(self, market_price, option_type='call', method='newton', 
                          initial_guess=0.3, max_iter=100, tolerance=1e-6):
        # Select pricing function
        if option_type.lower() == 'call':
            price_func = market_price
        elif option_type.lower() == 'put':
            price_func = market_price
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Objective function: difference between model and market price
        def objective(sigma):
            if sigma <= 0:
                return 1e10  # Penalty for negative volatility
            return price_func(sigma) - market_price
        
        try:
            if method == 'newton':
                # Newton-Raphson method (faster, needs good initial guess)
                iv = newton(objective, initial_guess, maxiter=max_iter, tol=tolerance)
            elif method == 'brentq':
                # Brent's method (more robust, no initial guess needed)
                # Search between 0.001% and 500% volatility
                iv = brentq(objective, 0.00001, 5.0, maxiter=max_iter, xtol=tolerance)
            else:
                raise ValueError("method must be 'newton' or 'brentq'")
            
            # Validate result
            if iv < 0 or iv > 5:
                return None
                
            return iv
        except (RuntimeError, ValueError) as e:
            print(f"Warning: IV calculation failed - {str(e)}")
            return None
        

def selection_condition(option_info, asset_market_price):
    bid_option_price = option_info.bid
    ask_option_price = option_info.ask
    option_market_price = (bid_option_price + ask_option_price)/2
    K = option_info.option_strike
    option_type = option_info.option_right  # 1 for call, 2 for put
    if option_info is None:
        print(f"Failed to get info for option {option_name}")
        return False
    if bid_option_price == 0.0 or ask_option_price == 0.0 or asset_market_price > K * (1 + STRIKE_PRICE_OFFSET) or asset_market_price < K * (1 - STRIKE_PRICE_OFFSET):
        return False                
    return True
        


    

if __name__ == "__main__":

    mt5_conn = MT5Connector()
    symbol_info = mt5_conn.get_symbol_info("BBAS3")
    selected=mt5_conn.symbol_select("BBAS3",True) 
    if not selected: 
        print("Failed to select BBAS3") 
        mt5_conn.shutdown() 
        quit() 
        mt5_conn.symbol_select("BBAS3",True)

    bid_market_price = symbol_info.bid
    ask_market_price = symbol_info.ask
    asset_market_price = (bid_market_price + ask_market_price)/2       
    
    print(f"Minimum time to expiration for BBAS* options {MIN_DAYS_TO_EXPIRY} seconds")
    
    time_now = int(time.time())
    minimum_exp_time = time_now + MIN_DAYS_TO_EXPIRY

    chain_options = mt5_conn.get_option_names_by_expiration_time("BBAS*")

    print(f"Names for the options after 10 days {chain_options[list(chain_options.keys())[0]]}")
    print(f"Minimum Expiration Time : {minimum_exp_time}")

    options_names_list = chain_options[list(chain_options.keys())[0]]
    expiration_time = list(chain_options.keys())[0]
    print(f"Listing options from the selected expiration time: {datetime.fromtimestamp(expiration_time)}")
    total_days = (expiration_time - time_now)/ UNIX_DAYS_IN_SECONDS
    print(f"T days : {total_days}")
    quant_calc = QuantCalculation()
    spot_prices_data = mt5_conn.get_data("BBAS3", mt5_conn.get_mt5_connector().TIMEFRAME_D1, 100, 0)["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100
    print(f"GARCH Volatility : {garch_vol:.2f}%")

    print(f"Asset Market Price : {asset_market_price:.2f}")   

    print("Asset BBAS3 ")

    black_scholes_calculator = BlackScholesCalculator()
   
    T = count_weekdays(datetime.fromtimestamp(time_now), int(total_days))
    
    # expiration_time is a Unix timestamp (seconds); convert to pandas Timestamp for interpolation
    r = black_scholes_calculator.get_interpolated_rate(pd.to_datetime(expiration_time, unit='s'))
    print(f"Futures Rate for {datetime.fromtimestamp(expiration_time).date()}: {r}")
    factor = (r/100+1)**((-T)/252)
    print(f"Factor : {factor}")
    F = asset_market_price / factor
    print(f"Fut DI Price : {F:.2f}")
    call_deltas_list = []
    put_deltas_list = []
    for option_name in options_names_list:
        option_info = mt5_conn.get_symbol_info(option_name)
        selected_option = mt5_conn.symbol_select(option_name,True)

        if not selected_option: 
            print(f"Failed to select option {option_name}") 
        else:
            option_info = mt5_conn.get_symbol_info(option_name)

            if option_info is None:
                print(f"Failed to get info for option {option_name}")
                continue
            if not selection_condition(option_info, asset_market_price):
                #print(f"Option {option_name} has no market data (bid and ask are zero). Skipping.")
                continue                
       
            bid_option_price = option_info.bid
            ask_option_price = option_info.ask
            option_market_price = (bid_option_price + ask_option_price)/2
            K = option_info.option_strike
            option_type = option_info.option_right  # 0 for call, 1 for put
               
            # Calculate implied volatility
            iv = black_scholes_calculator.fx_vol(F, K, T, option_market_price, factor, option_type)
            delta = black_scholes_calculator.fx_delta(F, K, T, iv, factor, option_type, asset_market_price)
            delta = round(delta, 2)
            option_type_str = "Call" if option_type == CALL_OPTION else "Put"
            iv = iv * 100  # Convert to percentage
            diff_vol = garch_vol - iv
            print(f"GARCH - IV for option {option_name} is {garch_vol:.2f}% : {iv:.2f}% diff is {diff_vol:.2f}% delta {delta:.2f}")

            if option_type == CALL_OPTION:  
                call_deltas_list.append(delta)
            elif option_type == PUT_OPTION:
                put_deltas_list.append(delta)

    if call_deltas_list:
        avg_call_delta = sum(call_deltas_list) / len(call_deltas_list)
        std_call_delta = (sum((x - avg_call_delta) ** 2 for x in call_deltas_list) / len(call_deltas_list)) ** 0.5
        
        # Find closest deltas to ±1 std from mean
        lower_bound = avg_call_delta - std_call_delta
        upper_bound = avg_call_delta + std_call_delta
        closest_lower = min(call_deltas_list, key=lambda x: abs(x - lower_bound))
        closest_upper = min(call_deltas_list, key=lambda x: abs(x - upper_bound))
        
        print(f"Call Deltas: {call_deltas_list}")
        print(f"Average Call Delta: {avg_call_delta:.2f} ± {std_call_delta:.2f}")
        print(f"Range: [{avg_call_delta - std_call_delta:.2f}, {avg_call_delta + std_call_delta:.2f}]")
        print(f"Closest existing deltas: Lower={closest_lower:.2f}, Upper={closest_upper:.2f}")
    else:
        print(f"Call Deltas: {call_deltas_list}")
        print("Average Call Delta: N/A (no data)")
       
    if put_deltas_list:
        avg_put_delta = sum(put_deltas_list) / len(put_deltas_list)
        std_put_delta = (sum((x - avg_put_delta) ** 2 for x in put_deltas_list) / len(put_deltas_list)) ** 0.5
        
        # Find closest deltas to ±1 std from mean
        lower_bound = avg_put_delta - std_put_delta
        upper_bound = avg_put_delta + std_put_delta
        closest_lower = min(put_deltas_list, key=lambda x: abs(x - lower_bound))
        closest_upper = min(put_deltas_list, key=lambda x: abs(x - upper_bound))
        
        print(f"Put Deltas: {put_deltas_list}")
        print(f"Average Put Delta: {avg_put_delta:.2f} ± {std_put_delta:.2f}")
        print(f"Range: [{avg_put_delta - std_put_delta:.2f}, {avg_put_delta + std_put_delta:.2f}]")
        print(f"Closest existing deltas: Lower={closest_lower:.2f}, Upper={closest_upper:.2f}")
    else:
        print(f"Put Deltas: {put_deltas_list}")
        print("Average Put Delta: N/A (no data)")
