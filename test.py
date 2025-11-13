import numpy as np
from scipy.stats import norm
from constants import CALL_OPTION
from mt5_connector import MT5Connector
import asyncio
from datetime import datetime
from scipy.optimize import newton, brentq

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



def get_options_names():
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return None, None
    symbol_y = "BBAS3"
    spot_data = mt5_conn.get_data(symbol_y)
    if spot_data is None:
        print("Failed to get historical data")
    else:
        print(f"Retrieved {spot_data.head(1)} for {symbol_y}")

    values = mt5_conn.get_options_chain("BBAS*")

    for option_symbol in values:
        symbol_info=mt5_conn.get_symbol_info(option_symbol)
        if symbol_info!=None:
           ask = symbol_info.ask
           bid = symbol_info.bid
           if ask > 0.0 and bid > 0.0:
               avg_price = (ask + bid) / 2
               print(f"Option: {option_symbol}, Bid: {bid:.2f}, Ask: {ask:.2f}, Avg: {avg_price:.2f}")
    return values, spot_data

# Example Usage
if __name__ == "__main__":

    mt5_conn = MT5Connector()
    underlying_symbol = "BOVA11"
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        exit()  

    spot_data = mt5_conn.get_symbol_info(underlying_symbol)
    spot_price = (spot_data.ask + spot_data.bid) / 2 if spot_data else None

    print(f"Current spot price for {underlying_symbol}: {spot_price}")
    if spot_data is None:
        print("Failed to get historical data")
        exit()
    else:
        print(f"Retrieved price {spot_price} for {underlying_symbol}") 

    
    data = mt5_conn.get_options_chain("PETR*", CALL_OPTION)
    print("\n" + "=" * 70)
    print("FETCHING OPTIONS CHAIN FOR PETR*")
    option_data = data[0] if data else None
    print(f"First Option Data: {option_data}")
    option_details = mt5_conn.get_symbol_info(option_data) if option_data else None
    print(f"Option Details: {option_details}")

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Option IV and Delta Calculation")
    print("=" * 70)

    # Market parameters
    S = spot_price         # Current stock price
    K = 33.5         # Strike price
    T = 0.11        # 28 dias to expiration
    r = 0.149        # 14.9% risk-free rate
    q = 0.0        # 0% dividend yield
    market_price = 2.56  # Observed call option price

    # Create an instance of the BlackScholesIV class
    bs_iv = BlackScholesIV(S=S, K=K, T=T, r=r, q=q)

    # Calculate implied volatility
    implied_vol = bs_iv.implied_volatility(market_price)
    delta = bs_iv.call_delta(implied_vol)
    vega = bs_iv.vega(implied_vol)

    print(f"Implied volatility: {implied_vol:.4f}, Delta: {delta:.4f}, Vega: {vega:.4f}")
