import numpy as np
from scipy.stats import norm
from mt5_connector import MT5Connector
import asyncio

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

async def get_prices():
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return None, None
    symbol_y = "BBAS3"
    spot_data = mt5_conn.get_data(symbol_y)
    if spot_data is None:
        print("Failed to get historical data")
    else:
        print(f"Retrieved {spot_data.head()} for {symbol_y}")

    option_data = mt5_conn.get_options_chain("BBAS*")
    if option_data is None:
        print("Failed to get historical data")
    else:
        for option in option_data:
            if option is not None:
                print(f"Retrieved {option_data[1][5]} for options chain")
                break

asyncio.run(get_prices())