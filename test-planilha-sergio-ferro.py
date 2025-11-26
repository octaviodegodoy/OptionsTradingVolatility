
from datetime import datetime
import time
from scipy.stats import norm
from scipy.optimize import newton, brentq
from constants import CALL_OPTION, MIN_DAYS_TO_EXPIRY, UNIX_DAYS_IN_SECONDS
from functions.quant_functions import QuantCalculation
from mt5_connector import MT5Connector
from functions.original_spline import c_spline
import pandas as pd

from utils import count_weekdays

class BlackScholesCalculator:
    def __init__(self):
        pass
    
    
    def fx_put(self, F, K, T, sigma, r):
        d1_value = self.d1(F, K, T, sigma)
        d2_value = self.d2(F, K, T, sigma)

        N_neg_d1 = norm.cdf(-d1_value)
        N_neg_d2 = norm.cdf(-d2_value)
        put_price = (K * N_neg_d2 - F * N_neg_d1) * r  # Assuming a risk-free rate of 5%
        
        if put_price < 0:
            put_price = 0.0

        return put_price
    
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
    

if __name__ == "__main__":

    mt5_conn = MT5Connector()
    symbol_info = mt5_conn.get_symbol_info("ABEV3")
    selected=mt5_conn.symbol_select("ABEV3",True) 
    if not selected: 
        print("Failed to select ABEV3") 
        mt5_conn.shutdown() 
        quit() 
        mt5_conn.symbol_select("ABEV3",True)

    bid_market_price = symbol_info.bid
    ask_market_price = symbol_info.ask
    asset_market_price = (bid_market_price + ask_market_price)/2    

    quant_calc = QuantCalculation()
    spot_prices_data = mt5_conn.get_data("BBAS3", mt5_conn.get_mt5_connector().TIMEFRAME_D1, 100, 0)["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data)
    print(f"GARCH Volatility : {garch_vol}")

    print(f"Asset Market Price : {asset_market_price}")

    # Obtain Call Option Names

    get_call_option_names = mt5_conn.get_call_option_name_list("BBAS*")
    print(f"Call Option Names : {get_call_option_names}")

    # Get interest rate
    data = mt5_conn.get_data("DI1@", mt5_conn.get_mt5_connector().TIMEFRAME_MN1, 7, 0)
    interest_rates = data.sort_index(ascending=True)
    print(f"Interest Rate : {interest_rates}")

    # Convert timestamps to numeric values (days since first date) for spline interpolation
    time_numeric = (interest_rates['time'] - interest_rates['time'].iloc[0]).dt.total_seconds() / 86400
    
    # Helper function to get interpolated rate for any future date
    def get_interpolated_rate(target_date):
        """Get interpolated interest rate for a specific date"""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        target_numeric = (target_date - interest_rates['time'].iloc[0]).total_seconds() / 86400
        
        rate = c_spline(
            x_data=time_numeric.values,
            y_data=interest_rates['close'].values,
            x_eval=target_numeric
        )
        return rate
    
    # Example: Get rate for specific date
    future_date = pd.to_datetime('2025-11-20')
    print(f"Interest Rate Time : {future_date}")
    print(f"Interest Rate Close (last known): {interest_rates['close'].iloc[-1]}")
    
    futures_rate = get_interpolated_rate(future_date)
    print(f"Futures Rate for {future_date.date()}: {futures_rate}")
    
    # You can now easily test different dates:
    # futures_rate_dec = get_interpolated_rate('2025-12-1')
    # futures_rate_30days = get_interpolated_rate(pd.Timestamp.now() + pd.DateOffset(days=30))

    # Fut DI
    T = 32
    factor = (futures_rate/100+1)**((-T)/252)
    print(f"Factor : {factor}")
    F = asset_market_price / factor
    print(f"Fut DI Price : {F}")

    print("Test BBAS3 ")
    
    print(f"Minimum time to expiration for BBAS* options {MIN_DAYS_TO_EXPIRY} seconds")
    
    time_now = int(time.time())
    minimum_exp_time = time_now + MIN_DAYS_TO_EXPIRY

    chain_options = mt5_conn.get_option_names_by_expiration_time("BBAS*")

    print(f"Names for the options after 10 days {chain_options[list(chain_options.keys())[0]]}")
    print(f"Minimum Expiration Time : {minimum_exp_time}")

    options_names_list = chain_options[list(chain_options.keys())[0]]
    print(f"Listing options from the selected expiration time: {datetime.fromtimestamp(list(chain_options.keys())[0])}")
    for option_name in options_names_list:
        option_info = mt5_conn.get_symbol_info(option_name)
        selected_option = mt5_conn.symbol_select(option_name,True)
        if not selected_option: 
            print(f"Failed to select option {option_name}") 
        else:
            option_info = mt5_conn.get_symbol_info(option_name)
            bid_option_price = option_info.bid
            ask_option_price = option_info.ask
            if option_info is None:
                print(f"Failed to get info for option {option_name}")
                continue
            if bid_option_price == 0.0 or ask_option_price == 0.0:
                #print(f"Option {option_name} has no market data (bid and ask are zero). Skipping.")
                continue                
       
            option_market_price = (bid_option_price + ask_option_price)/2    
        
            print(f"Option Name: {option_name}, Market Price: {option_market_price} bid {bid_option_price} ask {ask_option_price}")
        

    total_days = (list(chain_options.keys())[0] - time_now)/ UNIX_DAYS_IN_SECONDS
    print(f"T days : {total_days}")

    T = count_weekdays(datetime.fromtimestamp(time_now), int(total_days))

    print(f"The tenor variable is {T} days")       


