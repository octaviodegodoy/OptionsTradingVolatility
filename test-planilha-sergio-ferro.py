
from scipy.stats import norm
from scipy.optimize import newton, brentq
from constants import CALL_OPTION
from functions.quant_functions import QuantCalculation
from mt5_connector import MT5Connector

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

    print(f"Asset Market Price : {asset_market_price}")

    quant_calc = QuantCalculation()
    garch_vol = quant_calc.agarch_estimation(mt5_conn.get_data("ABEV3")["close"].values)
    print(f"GARCH Volatility : {garch_vol}")

    get_call_option_names = mt5_conn.get_call_option_name_list("ABEV*")
    print(f"Call Option Names : {get_call_option_names}")

    chain_list = mt5_conn.get_options_chain("ABEV*", CALL_OPTION)
    print(f"Option Chain DataFrame : {chain_list}")



