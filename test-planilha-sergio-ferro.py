from ast import If
from math import log
from datetime import datetime
import time
from scipy.stats import norm
from scipy.optimize import newton, brentq
from constants import CALL_OPTION, MIN_DAYS_TO_EXPIRY, PUT_OPTION, STRIKE_PRICE_OFFSET, TYPE_BUY, UNIX_DAYS_IN_SECONDS
from functions.quant_functions import QuantCalculation
from mt5_connector import MT5Connector
from functions.original_spline import c_spline
import pandas as pd
import numpy as np

from utils import count_weekdays

class BlackScholesCalculator:
    def __init__(self):
        pass

    def d_1(self, F, K, T, sigma):
        return (log(F / K) + sigma ** 2 * (T / 252) / 2) / (sigma * (T / 252) ** (1 / 2))
    
    def d_2(self, F, K, T, sigma):
        return self.d_1(F, K, T, sigma) - sigma * (T / 252) ** (1 / 2)
    
    def black_scholes_call(self, F, K, T, sigma, r):
        """European call option price using Black-Scholes forward formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * (T / 252)) / (sigma * np.sqrt(T / 252))
        d2 = d1 - sigma * np.sqrt(T / 252)
        
        # Using forward price formula with discount factor
        call_price = r * (F * norm.cdf(d1) - K * norm.cdf(d2))
        
        return max(call_price, 0.0)

    def black_scholes_put(self, F, K, T, sigma, r):
        """European put option price using Black-Scholes forward formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * (T / 252)) / (sigma * np.sqrt(T / 252))
        d2 = d1 - sigma * np.sqrt(T / 252)
        
        # Using forward price formula with discount factor
        put_price = r * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        return max(put_price, 0.0)

    def implied_vol(self, F, K, T, price, r, option_type):
        """
        Calculate implied volatility using Brent's method.
        
        Args:
            F: Forward price (spot / discount_factor)
            K: Strike price
            T: Time to expiration in trading days
            price: Market price of the option
            r: Discount factor (not the rate itself)
            option_type: 0 for call, 1 for put
        
        Returns:
            Implied volatility as decimal (e.g., 0.20 for 20%)
        """
        if price <= 0 or T <= 0:
            return np.nan
        
        def objective(sigma):
            if sigma <= 0:
                return 1e10
            
            if option_type == CALL_OPTION:  # Call option
                theoretical_price = self.black_scholes_call(F, K, T, sigma, r)
            elif option_type == PUT_OPTION:  # Put option
                theoretical_price = self.black_scholes_put(F, K, T, sigma, r)
            else:
                return 1e10
            
            return theoretical_price - price
        
        try:
            # Search sigma between 0.1% and 500%
            iv = brentq(objective, 0.001, 5.0, maxiter=100)
            return iv
        except (ValueError, RuntimeError) as e:
            # If brentq fails, try with wider bounds or return NaN
            try:
                iv = brentq(objective, 0.0001, 10.0, maxiter=200)
                return iv
            except:
                return np.nan
    

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

    while selected:

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

        positions = mt5_conn.get_open_positions()
        print(f"Total open positions: {len(positions)}")

        black_scholes_calculator = BlackScholesCalculator()

        T = count_weekdays(datetime.fromtimestamp(time_now), int(total_days))

        # expiration_time is a Unix timestamp (seconds); convert to pandas Timestamp for interpolation
        r = black_scholes_calculator.get_interpolated_rate(pd.to_datetime(expiration_time, unit='s'))
        print(f"Futures Rate for {datetime.fromtimestamp(expiration_time).date()}: {r}")
        factor = (r/100+1)**((-T)/252)
        print(f"Factor : {factor}")
        F = asset_market_price / factor
        print(f"Fut DI Price : {F:.2f}")
        call_deltas_dict = {}
        put_deltas_dict = {}  # Dictionary to store {delta: iv}
        call_deltas_list = []
        call_iv_list = []
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
                iv_call_brentq = black_scholes_calculator.implied_vol(F, K, T, option_market_price, factor, option_type)
                print(f"Option {option_name} IV {iv_call_brentq}")
                iv = black_scholes_calculator.fx_vol(F, K, T, option_market_price, factor, option_type)
                delta = black_scholes_calculator.fx_delta(F, K, T, iv, factor, option_type, asset_market_price)
                delta = round(delta, 2)
                option_type_str = "Call" if option_type == CALL_OPTION else "Put"
                iv = iv * 100  # Convert to percentage
                diff_vol = garch_vol - iv
                print(f"GARCH - IV for option {option_name} is {garch_vol:.2f}% : {iv:.2f}% diff is {diff_vol:.2f}% delta {delta:.2f}")

                if option_type == CALL_OPTION:
                    call_deltas_dict[delta] = {'iv': round(iv, 2), 'option_name': option_name}
                elif option_type == PUT_OPTION:
                    put_deltas_dict[delta] = {'iv': round(iv, 2), 'option_name': option_name}


        if call_deltas_dict:
            deltas = list(call_deltas_dict.keys())
            avg_call_delta = sum(deltas) / len(deltas)
            std_call_delta = (sum((x - avg_call_delta) ** 2 for x in deltas) / len(deltas)) ** 0.5
            
            # Find closest deltas to ±1 std from mean
            lower_bound = avg_call_delta - std_call_delta
            upper_bound = avg_call_delta + std_call_delta
            closest_lower = min([d for d in deltas if d > 0.25], key=lambda x: abs(x - lower_bound), default=None)
            closest_upper = min([d for d in deltas if d < 0.75], key=lambda x: abs(x - upper_bound), default=None)
            
            # Calculate IV difference
            if closest_lower is not None and closest_upper is not None:
                iv_diff = call_deltas_dict[closest_upper]['iv'] - call_deltas_dict[closest_lower]['iv']
                call_delta_diff = closest_upper - closest_lower
            else:
                iv_diff = None

         
            print(f"Call Deltas: {deltas}")
            print(f"Average Call Delta: {avg_call_delta:.2f} ± {std_call_delta:.2f} delta diff {call_delta_diff:.2f}")
            print(f"Closest existing deltas: Lower={closest_lower:.2f} option name {call_deltas_dict[closest_lower]['option_name']} with (IV={call_deltas_dict[closest_lower]['iv']:.2f}%), Upper={closest_upper:.2f} and option name {call_deltas_dict[closest_upper]['option_name']}  with (IV={call_deltas_dict[closest_upper]['iv']:.2f}%) diff IV={iv_diff:.2f}%" if iv_diff is not None else "N/A  (IV difference)")

            
            call_buy = call_deltas_dict[closest_upper]['option_name']
            call_sell = call_deltas_dict[closest_lower]['option_name']
            orders_type = [mt5_conn.ORDER_TYPE_SELL, mt5_conn.ORDER_TYPE_BUY] # or "SELL"
            iv_y = call_deltas_dict[closest_upper]['iv']
            iv_x = call_deltas_dict[closest_lower]['iv']
            if iv_diff is not None and iv_diff < 0.005:
                 print(f"Placing vertical spread order for call options: Buy {call_buy}, Sell {call_sell}, IV difference {iv_diff:.2f}%")
                 mt5_conn.place_order_vertical(call_buy, call_sell, orders_type, 100, iv_y, iv_x)

            
            positions = mt5_conn.get_open_positions()
            print(f"Total open positions: {len(positions)}")
            asset_symbol = "BBAS3"
            sum_delta_calls = 0.0
            call_buy_strike = None
            
            for pos in positions:
                print(f"Analyzing position: {pos.symbol}")
                option_bid_price = mt5_conn.get_symbol_info(pos.symbol).bid
                option_ask_price = mt5_conn.get_symbol_info(pos.symbol).ask
                option_market_price = (option_bid_price + option_ask_price)/2 
                
                
                asset_bid_price = mt5_conn.get_symbol_info(asset_symbol).bid
                asset_ask_price = mt5_conn.get_symbol_info(asset_symbol).ask
                asset_market_price = (asset_bid_price + asset_ask_price)/2

                symbol_info = mt5_conn.get_symbol_info(pos.symbol)
                
                position_expiration_time = symbol_info.expiration_time
                print(f"Listing options from the selected expiration time: {datetime.fromtimestamp(position_expiration_time)}")
                r = black_scholes_calculator.get_interpolated_rate(pd.to_datetime(position_expiration_time, unit='s'))
                total_days_left = (position_expiration_time - time_now)/ UNIX_DAYS_IN_SECONDS

                t = count_weekdays(datetime.fromtimestamp(time_now), int(total_days_left))
                factor = (r/100+1)**((-t)/252)
                F = asset_market_price / factor

                K = symbol_info.option_strike
                option_type = symbol_info.option_right  # 0 for call, 1 for put
                iv = black_scholes_calculator.fx_vol(F, K, t, option_market_price, factor, option_type)
                delta = black_scholes_calculator.fx_delta(F, K, t, iv, factor, option_type, asset_market_price)
                delta = round(delta, 2) if pos.type == 0 else round(-delta, 2)
                call_buy_strike = K if pos.type == TYPE_BUY else None
                if option_type == CALL_OPTION:
                    sum_delta_calls += delta * pos.volume
                print(f"Position: {pos.symbol}, Delta: {delta:.2f} position type {pos.type} volume {pos.volume} open price {pos.price_open}")
                time.sleep(1)
            
            
            
            print(f"Get delta from put option strike {call_buy_strike}")
            put_option_name = mt5_conn.get_option_name_by_strike("BBAS*", call_buy_strike, PUT_OPTION, position_expiration_time)
            print(f"Put option strike found {put_option_name}")
            # calculate put option IV and delta
            put_option_info = mt5_conn.get_symbol_info(put_option_name)
            selected = mt5_conn.symbol_select(put_option_name,True)
            if not selected: 
                print(f"Failed to select option {put_option_name}")
            else:
                put_option_info = mt5_conn.get_symbol_info(put_option_name)
                put_bid_option_price = put_option_info.bid
                put_ask_option_price = put_option_info.ask
                print(f"Put Bid {put_bid_option_price}  Put Ask {put_ask_option_price}")
                put_option_market_price = (put_bid_option_price + put_ask_option_price)/2
                K_put = put_option_info.option_strike
                option_type_put = put_option_info.option_right  # 0 for call, 1
                iv_put = black_scholes_calculator.fx_vol(F, K_put, t, put_option_market_price, factor, option_type_put)
                delta_put = black_scholes_calculator.fx_delta(F, K_put, t, iv_put, factor, option_type_put, asset_market_price)
                delta_put = round(delta_put, 2)
                put_delta_ratio = sum_delta_calls / delta_put if delta_put != 0 else None
                hedge_volume = round(abs(put_delta_ratio) / 100) * 100
                print(f"Total open positions: {len(positions)} sum delta call {abs(sum_delta_calls):.2f} delta put is {delta_put:.2f} expected delta puts to hedge {-put_delta_ratio:.2f} rounded to {-hedge_volume}")



 
            
        else:
            print(f"Call Deltas: {list(call_deltas_dict.keys())}")
            print("Average Call Delta: N/A (no data)")
            
        if put_deltas_dict:
            deltas = list(put_deltas_dict.keys())
            avg_put_delta = sum(deltas) / len(deltas)
            std_put_delta = (sum((x - avg_put_delta) ** 2 for x in deltas) / len(deltas)) ** 0.5
            
            # Find closest deltas to ±1 std from mean
            lower_bound = avg_put_delta - std_put_delta
            upper_bound = avg_put_delta + std_put_delta
            print(f" Put lower and upper bounds {lower_bound}  :   {upper_bound}")
            closest_lower = min([d for d in deltas if -0.75 < d < -0.25], key=lambda x: abs(x - lower_bound), default=None)
            closest_upper = min([d for d in deltas if -0.75 < d < -0.25], key=lambda x: abs(x - upper_bound), default=None)
            print(f" lower and upper {closest_lower}  :   {closest_upper}")
            put_delta_diff = None
            iv_diff = None
            # Calculate IV difference
            if closest_lower is not None and closest_upper is not None:
                iv_diff = put_deltas_dict[closest_upper]['iv'] - put_deltas_dict[closest_lower]['iv']
                put_delta_diff = closest_upper - closest_lower
            else:
                iv_diff = None
            
            print(f"Put Deltas: {deltas}")
            print(f"Average Put Delta: {avg_put_delta:.2f} ± {std_put_delta:.2f} delta diff {put_delta_diff:.2f}")
            print(f"Closest existing deltas: Lower={closest_lower:.2f} option name {put_deltas_dict[closest_lower]['option_name']} with (IV={put_deltas_dict[closest_lower]['iv']:.2f}%), Upper={closest_upper:.2f} and option name {put_deltas_dict[closest_upper]['option_name']} (IV={put_deltas_dict[closest_upper]['iv']:.2f}%) diff IV={iv_diff:.2f}%" if iv_diff is not None else "N/A  (IV difference)")
            
        else:
            print(f"Put Deltas: {list(put_deltas_dict.keys())}")
            print("Average Put Delta: N/A (no data)")


        time.sleep(10)  # Wait for 60 seconds before the next iteration
