import numpy as np
from scipy.stats import norm
from functions.black_scholes import BlackScholesCalculator
from constants import ASSET_SYMBOL, CALL_OPTION, GARCH_SAMPLE_SIZE, PERIODS
from mt5_connector import MT5Connector
import asyncio
import time
from scipy.optimize import newton, brentq
from functions.quant_functions import QuantCalculation
from utils import Utils

class BlackScholesIV:
    """
    Black-Scholes Implied Volatility and Greeks Calculator
    Includes dividend yield (q) in the model
    """
    
    def __init__(self, S, K, T, r, q=0):
        """
        Initialize with market parameters
        
        Parameters:
        S: Current stock/underlying price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        q: Dividend yield (annual, continuous)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.utils = Utils()
        
    def _d1(self, sigma):
        """Calculate d1 parameter"""
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
    
    def _d2(self, sigma):
        """Calculate d2 parameter"""
        return self._d1(sigma) - sigma * np.sqrt(self.T)
    
    def call_price(self, sigma):
        """Calculate Black-Scholes call option price"""
        d1 = self._d1(sigma)
        d2 = self._d2(sigma)
        
        call = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        return call
    
    def put_price(self, sigma):
        """Calculate Black-Scholes put option price"""
        d1 = self._d1(sigma)
        d2 = self._d2(sigma)
        
        put = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
               self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        return put
    
    def vega(self, sigma):
        """Calculate option vega (same for calls and puts)"""
        d1 = self._d1(sigma)
        vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
        return vega
    
    def call_delta(self, sigma):
        """Calculate call option delta"""
        d1 = self._d1(sigma)
        return np.exp(-self.q * self.T) * norm.cdf(d1)
    
    def put_delta(self, sigma):
        """Calculate put option delta"""
        d1 = self._d1(sigma)
        return -np.exp(-self.q * self.T) * norm.cdf(-d1)




    # Black-Scholes FX Option Pricing Functions planilha Sergio Ferro
    def d_1(forward, strike, tenor, sigma):
        """
        Calculate d1 parameter for Black-Scholes option pricing model
        
        Parameters:
        forward (float): Forward price
        strike (float): Strike price
        tenor (float): Time to expiration in days
        sigma (float): Volatility (annualized)
        
        Returns:
        float: d1 parameter
        """
        time_to_expiry = tenor / 252  # Convert days to years (252 trading days)
        d_1 = (np.log(forward / strike) + sigma**2 * time_to_expiry / 2) / (sigma * np.sqrt(time_to_expiry))
        return d_1

    
    def d_2(forward, strike, tenor, sigma):
        """
        Calculate d2 parameter for Black-Scholes option pricing model
        
        Parameters:
        forward (float): Forward price
        strike (float): Strike price
        tenor (float): Time to expiration in days
        sigma (float): Volatility (annualized)
        
        Returns:
        float: d2 parameter
        """
        time_to_expiry = tenor / 252  # Convert days to years (252 trading days)
        d_2 = d_1(forward, strike, tenor, sigma) - sigma * np.sqrt(time_to_expiry)
        return d_2

    def fx_call(forward, strike, tenor, sigma, interest):
        """
        Calculate FX call option price using Black-Scholes model
        
        Parameters:
        forward (float): Forward price
        strike (float): Strike price
        tenor (float): Time to expiration in days
        sigma (float): Volatility (annualized)
        interest (float): Discount factor (e.g., e^(-r*T))
        
        Returns:
        float: FX call option price
        """
        x = d_1(forward, strike, tenor, sigma)
        y = d_2(forward, strike, tenor, sigma)
        fx_call = (forward * norm.cdf(x) - strike * norm.cdf(y)) * interest
        return fx_call

    def fx_call_vol(forward, strike, tenor, price, interest):
        """
        Calculate implied volatility for FX call option using binary search
        
        Parameters:
        forward (float): Forward price
        strike (float): Strike price
        tenor (float): Time to expiration in days
        price (float): Market price of the option
        interest (float): Discount factor (e.g., e^(-r*T))
        
        Returns:
        float: Implied volatility
        """
        high = 5.0  # Upper bound for volatility search
        low = 0.0   # Lower bound for volatility search
        
        # Binary search for implied volatility
        while (high - low) > 0.00000001:
            mid_vol = (high + low) / 2
            if fx_call(forward, strike, tenor, mid_vol, interest) > price:
                high = mid_vol
            else:
                low = mid_vol
        
        return (high + low) / 2


    def test_fx_option_pricing():
        """
        Test FX option pricing functions
        """
        # Example: Calculate implied volatility from market option price
        forward_price = 28.45
        strike_price = 28
        time_to_expiry_days = 30
        market_option_price = 6.50  # Market price of the call option
        discount_factor = 0.99  # Discount factor (e^(-r*T))

        # STEP 1: Calculate implied volatility from market price
        print("STEP 1: Calculate Implied Volatility from Market Price")
        print(f"Forward price: {forward_price}")
        print(f"Strike price: {strike_price}")
        print(f"Time to expiry: {time_to_expiry_days} days")
        print(f"Market option price: {market_option_price:.4f}")
        print(f"Discount factor: {discount_factor}")
        print("-" * 50)
        
        implied_vol = fx_call_vol(forward_price, strike_price, time_to_expiry_days, market_option_price, discount_factor)
        
        print(f"Implied Volatility: {implied_vol:.4f} ({implied_vol*100:.2f}%)")
        print("-" * 50)
        
        # STEP 2: Calculate d1 and d2 using the implied volatility
        print("\nSTEP 2: Calculate Black-Scholes Parameters")
        d1_value = d_1(forward_price, strike_price, time_to_expiry_days, implied_vol)
        d2_value = d_2(forward_price, strike_price, time_to_expiry_days, implied_vol)
        
        print(f"d1 value: {d1_value:.4f}")
        print(f"d2 value: {d2_value:.4f}")
        
        # STEP 3: Verify by calculating option price with implied volatility
        print("\nSTEP 3: Verification")
        calculated_price = fx_call(forward_price, strike_price, time_to_expiry_days, implied_vol, discount_factor)
        print(f"Calculated option price: {calculated_price:.4f}")
        print(f"Market option price: {market_option_price:.4f}")
        print(f"Price difference: {abs(calculated_price - market_option_price):.8f}")

    def implied_volatility(self, market_price, option_type='call', method='newton', 
                            initial_guess=0.3, max_iter=100, tolerance=1e-6):
        # Select pricing function
            if option_type.lower() == 'call':
                price_func = self.call_price
            elif option_type.lower() == 'put':
                price_func = self.put_price
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

async def test_send_order():
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return
    
    chain_options = mt5_conn.get_option_names_by_expiration_time("BBAS*")
    options_names_list = chain_options[list(chain_options.keys())[0]]

    print(f"Names for the options after 10 days {options_names_list}")

    symbol_y = 'BBASL215W4'  # Example option symbol Y
    symbol_x = 'BBASL225W4'  # Example option symbol X
  #  symbol_selected=mt5_conn.get_symbol_info(symbol_y)
    selected_option = mt5_conn.symbol_select(symbol_y,True)
    if selected_option is None:
        print(f"Failed to get symbol info for {symbol_y}")
        return
    orders_type = [mt5_conn.ORDER_TYPE_BUY, mt5_conn.ORDER_TYPE_SELL]
    volume = 100  # Example volume
    iv_y = 0.25  # Example IV for symbol Y
    iv_x = 0.20  # Example IV for symbol X

    mt5_conn.place_order_vertical(symbol_y, symbol_x, orders_type, volume, iv_y, iv_x)

async def place_order_test():
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return
    orders_type = [mt5_conn.ORDER_TYPE_BUY, mt5_conn.ORDER_TYPE_SELL] # or "SELL"
    call_buy = 'BBASL215W4'
    call_sell = 'BBASL225W4'
    iv_y = 0.25  # Example IV for symbol Y
    iv_x = 0.20  # Example IV for symbol X
    iv_diff = 0.25  # Example IV for symbol Y
    volume = 1000.0  # Example volume
    print(f"Preparing to place vertical spread iv diff is {iv_diff}, and {0.3} is the minimum")
    mt5_conn.place_order_vertical(call_buy, call_sell, orders_type, volume, iv_y, iv_x)

async def close_all_positions_test():
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return  
    mt5_conn.close_all_positions()

async def get_options_names():
    mt5_conn = MT5Connector()
    utils = Utils()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return None
    chain_options = mt5_conn.get_option_names_by_expiration_time(ASSET_SYMBOL[2])
    print(f"Options names for {ASSET_SYMBOL[2]}: {chain_options}")
    expiration_time = list(chain_options.keys())[0]
    print(f"Expiration time for options: {time.fromtimestamp(expiration_time)}")
    selected_asset = mt5_conn.symbol_select(ASSET_SYMBOL[2],True)
    if not selected_asset:
        print(f"Failed to select {ASSET_SYMBOL[2]}")
        return None
    symbol_info = mt5_conn.get_symbol_info(ASSET_SYMBOL[2])

    if symbol_info is None:
        print(f"Failed to get symbol info for {ASSET_SYMBOL[2]}")
        return None
    calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
    print(f"Calls data: {calls_dict}")
    print(f"Puts data: {puts_dict}")

async def get_total_put_deltas():
    mt5_conn = MT5Connector()
    utils = Utils()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return None
    total_deltas = utils.get_total_put_deltas()
    print(f"Total put deltas from open positions: {total_deltas:.2f}")

async def compare_garch_iv_with_puts():
    mt5_conn = MT5Connector()
    quant_calc = QuantCalculation()
    black_scholes_calculator = BlackScholesCalculator()
    utils = Utils()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return None
    symbol_info = mt5_conn.get_symbol_info(ASSET_SYMBOL[2])
    asset_market_price = (symbol_info.bid + symbol_info.ask) / 2
    print(f"Symbol info for {ASSET_SYMBOL[2]}: {symbol_info.name}")
    spot_prices_data = mt5_conn.get_data(ASSET_SYMBOL[2], mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0)["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100
    print(f"GARCH Volatility : {garch_vol:.2f}%")
    iv_condition = False
    while not iv_condition:
       option_name = "PETRC420"
       selected_option = mt5_conn.symbol_select(option_name,True)
       if not selected_option:
           print(f"Failed to select option {option_name}")
           return None
       option_info = mt5_conn.get_symbol_info(option_name)
       if option_info is None or option_info.bid == 0.0 or option_info.ask == 0.0:
           print(f"Failed to get valid market data for option {option_name}")
           return None
       print(f"Option info for {option_name}: {option_info.name}, bid: {option_info.bid}, ask: {option_info.ask}")
       option_market_price = (option_info.bid + option_info.ask)/2
       K = option_info.option_strike
       option_type = option_info.option_right  # 0 for call, 1 for put
       expiration_time = option_info.expiration_time
       factor,T = utils.get_factor_from_expiration_time(expiration_time)
       F = asset_market_price / factor
       iv_brentq = black_scholes_calculator.implied_vol(F, K, T, option_market_price, factor, option_type)
       iv_brentq = iv_brentq*100  # Convert to percentage
       iv_garch_diff = garch_vol - iv_brentq
       iv_condition = iv_garch_diff > 0       
       print(f"Implied Volatility for {option_name}: {iv_brentq:.2f}%")
       if iv_brentq is not None:
           print(f"Difference between GARCH IV and implied volatility for {option_name}: {iv_garch_diff:.2f}%")
       if iv_condition:
           print(f"GARCH IV is higher than implied volatility for {option_name} by {iv_garch_diff:.2f}%")
           mt5_conn.place_order(option_name,MT5Connector.ORDER_TYPE_BUY, 500.0, symbol_info.ask, 10, str(iv_brentq))
       time.sleep(15)


asyncio.run(compare_garch_iv_with_puts())