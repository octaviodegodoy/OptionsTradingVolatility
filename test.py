import numpy as np
from scipy.stats import norm
from functions.black_scholes import BlackScholesCalculator
from constants import (
    ASSET_SYMBOL, CALL_OPTION, PUT_OPTION, GARCH_SAMPLE_SIZE, PERIODS,
    PUT_SPREAD_EXPIRY_RANK,
    PUT_SPREAD_LONG_DELTA_MIN, PUT_SPREAD_LONG_DELTA_MAX,
    PUT_SPREAD_SHORT_DELTA_MIN, PUT_SPREAD_SHORT_DELTA_MAX,
    PUT_SPREAD_MIN_IV_EDGE, PUT_SPREAD_CALL_WALL_OFFSET,
    SHORT_CALL_BUTTERFLY_EXPIRY_RANK,
    SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT,
    SHORT_CALL_BUTTERFLY_MIN_IV_EDGE,
    SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT,
    STRADDLE_MAX_DELTA_IMBALANCE, STRADDLE_ENTRY_MAX_NET_IV,
    TARGET_OPTION_EXPIRY_RANK,
    SHORT_STRANGLE_EXPIRY_RANK,
    SHORT_STRANGLE_CALL_DELTA_MIN, SHORT_STRANGLE_CALL_DELTA_MAX,
    SHORT_STRANGLE_PUT_DELTA_MIN, SHORT_STRANGLE_PUT_DELTA_MAX,
    SHORT_STRANGLE_MIN_IV_EDGE,
    SHORT_STRANGLE_MAX_DELTA_IMBALANCE,
    SHORT_STRANGLE_MIN_NET_CREDIT,
    LONG_CALL_BUTTERFLY_EXPIRY_RANK,
    LONG_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT,
    LONG_CALL_BUTTERFLY_MIN_BODY_RICHNESS,
    LONG_CALL_BUTTERFLY_MIN_REWARD_RISK,
    IRON_CONDOR_EXPIRY_RANK,
    IRON_CONDOR_SHORT_DELTA_MIN, IRON_CONDOR_SHORT_DELTA_MAX,
    IRON_CONDOR_MIN_WING_WIDTH,
    IRON_CONDOR_MIN_IV_EDGE,
    IRON_CONDOR_MIN_POP,
    IRON_CONDOR_MIN_REWARD_RISK,
    IRON_CONDOR_MAX_DELTA_IMBALANCE,
    SKEW_EXPIRY_RANK,
    SKEW_RR_DELTA, SKEW_NEAR_DELTA, SKEW_FAR_DELTA,
    SKEW_DELTA_TOLERANCE,
    SKEW_MIN_RR_PP,
    SKEW_MIN_SLOPE_PP_PER_DELTA,
    FLYAGONAL_BWB_EXPIRY_RANK,
    FLYAGONAL_DIAG_LONG_EXPIRY_RANK,
    FLYAGONAL_BWB_BODY_DELTA_MIN, FLYAGONAL_BWB_BODY_DELTA_MAX,
    FLYAGONAL_BWB_LOWER_DELTA_MIN, FLYAGONAL_BWB_LOWER_DELTA_MAX,
    FLYAGONAL_BWB_BODY_DIST_MIN_PCT, FLYAGONAL_BWB_BODY_DIST_MAX_PCT,
    FLYAGONAL_BWB_BROKEN_WING_RATIO,
    FLYAGONAL_PUT_DELTA_MIN, FLYAGONAL_PUT_DELTA_MAX,
    FLYAGONAL_PUT_DIST_MIN_PCT, FLYAGONAL_PUT_DIST_MAX_PCT,
    FLYAGONAL_MAX_NET_DEBIT_PCT,
    FLYAGONAL_MIN_THETA_TO_DEBIT,
    FLYAGONAL_MAX_VEGA_RATIO,
    FLYAGONAL_MAX_NET_DELTA_ABS,
    FLYAGONAL_MIN_REWARD_RISK,
    FLYAGONAL_IV_RANK_PROXY_MAX,
)
from functions.gamma_exposure_calc import bs_gamma, implied_vol_newton, infer_right_from_symbol_name
from mt5_connector import MT5Connector
import asyncio
import time
from datetime import datetime, timezone
from scipy.optimize import newton, brentq
from functions.quant_functions import QuantCalculation
from utils import Utils
import math

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


async def unselect_asset(symbol: str):
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return
    count = mt5_conn.unselect_options_by_underlying(symbol)
    if count > 0:
        print(f"Successfully unselected {count} option symbols for {symbol} from Market Watch")
    else:
        print(f"No selected option symbols found for {symbol} in Market Watch")


async def select_options_near_spot(symbol: str, expiry_rank: int = 1):
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return
    count = mt5_conn.select_options_near_spot(symbol, expiry_rank)
    if count > 0:
        print(f"Selected {count} option symbols for {symbol} (expiry rank {expiry_rank}) in Market Watch")
    else:
        print(f"No new option symbols to select for {symbol} (expiry rank {expiry_rank})")


def compute_call_wall(
    mt5_raw,
    underlying_basis: str,
    spot: float,
    risk_free_rate: float = 0.12,
    div_yield: float = 0.00,
) -> dict:
    """
    Compute call-only GEX by strike and return the call wall.

    The call wall is the strike with the highest aggregate call dollar-gamma
    (positive GEX). When OI (session_interest) is zero for all series the
    function falls back to session_volume as a proxy and warns.

    Returns a dict:
      {
        'call_wall': float,
        'gex_by_strike': dict[float, float],  # strike -> call GEX
        'oi_proxy': str,                       # 'session_interest' or 'session_volume'
        'total_call_gex': float,
      }
    Returns None when no options are found.
    """
    glob = underlying_basis.rstrip("0123456789") + "*"
    syms = mt5_raw.symbols_get(glob) or []
    now_epoch = int(datetime.now(timezone.utc).timestamp())

    gex_by_strike: dict = {}
    any_oi = False
    rows = []

    for s in syms:
        info = mt5_raw.symbol_info(s.name)
        if not info or getattr(info, "basis", "") != underlying_basis:
            continue
        if info.option_strike <= 0 or info.expiration_time <= 0:
            continue
        if info.expiration_time <= now_epoch:
            continue

        # Only calls contribute to the call wall
        right = None
        opt_right = getattr(info, "option_right", 0)
        if opt_right == 1:
            right = "C"
        else:
            right = infer_right_from_symbol_name(info.name, underlying_basis)
        if right != "C":
            continue

        mid = 0.0
        if info.bid > 0 and info.ask > 0:
            mid = (info.bid + info.ask) / 2.0
        elif getattr(info, "price_theoretical", 0.0):
            mid = float(info.price_theoretical)
        if mid <= 0:
            continue

        T_years = max(0.0, info.expiration_time - now_epoch) / (365.0 * 24 * 3600)
        if T_years <= 0:
            continue

        oi = float(getattr(info, "session_interest", 0.0) or 0.0)
        vol = float(getattr(info, "session_volume", 0.0) or 0.0)
        if oi > 0:
            any_oi = True

        rows.append({
            "strike": float(info.option_strike),
            "T": T_years,
            "mid": mid,
            "oi": oi,
            "vol": vol,
            "mult": float(info.trade_contract_size or 1.0),
        })

    if not rows:
        return None

    oi_proxy = "session_interest" if any_oi else "session_volume"
    if not any_oi:
        print("  [GEX] WARNING: session_interest=0 for all calls; falling back to session_volume as OI proxy.")

    for row in rows:
        sigma = implied_vol_newton(
            spot, row["strike"], row["T"],
            risk_free_rate, div_yield, "C", row["mid"]
        )
        if not sigma:
            continue
        g = bs_gamma(spot, row["strike"], row["T"], risk_free_rate, div_yield, sigma)
        oi_val = row["oi"] if any_oi else row["vol"]
        dollar_gamma = (spot ** 2) * g * row["mult"]
        call_gex = dollar_gamma * oi_val
        gex_by_strike[row["strike"]] = gex_by_strike.get(row["strike"], 0.0) + call_gex

    if not gex_by_strike:
        return None

    call_wall = max(gex_by_strike, key=lambda k: gex_by_strike[k])
    return {
        "call_wall": call_wall,
        "gex_by_strike": gex_by_strike,
        "oi_proxy": oi_proxy,
        "total_call_gex": sum(gex_by_strike.values()),
    }


def build_short_call_butterfly_candidates(utils, calls_dict, spot_price, garch_vol):
    call_by_strike = {}
    for call_delta, call_data in calls_dict.items():
        strike = call_data["strike"]
        current = call_by_strike.get(strike)
        if current is None or abs(call_delta - 0.50) < abs(current[0] - 0.50):
            call_by_strike[strike] = (call_delta, call_data)

    if len(call_by_strike) < 3:
        return []

    quote_by_strike = {}
    for strike, (_, call_data) in call_by_strike.items():
        option_info = utils.get_option_info_with_quote(call_data["option_name"])
        if option_info is None or option_info.bid <= 0.0 or option_info.ask <= 0.0:
            continue
        quote_by_strike[strike] = option_info

    strikes = sorted(quote_by_strike.keys())
    strikes_set = set(strikes)
    candidates = []

    for middle_idx in range(1, len(strikes) - 1):
        middle_strike = strikes[middle_idx]
        body_distance_pct = abs(middle_strike - spot_price) / max(spot_price, 1.0)
        if body_distance_pct > SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT:
            continue

        for lower_strike in strikes[:middle_idx]:
            wing_width = middle_strike - lower_strike
            if wing_width <= 0:
                continue

            upper_strike = round(middle_strike + wing_width, 8)
            if upper_strike not in strikes_set:
                continue

            lower_delta, lower_data = call_by_strike[lower_strike]
            middle_delta, middle_data = call_by_strike[middle_strike]
            upper_delta, upper_data = call_by_strike[upper_strike]
            lower_quote = quote_by_strike[lower_strike]
            middle_quote = quote_by_strike[middle_strike]
            upper_quote = quote_by_strike[upper_strike]

            net_credit = lower_quote.bid + upper_quote.bid - (2 * middle_quote.ask)
            max_profit = net_credit
            max_loss = wing_width - net_credit
            reward_risk = max_profit / max_loss if max_profit > 0 and max_loss > 0 else -math.inf
            body_iv_edge = garch_vol - middle_data["iv"]
            wing_richness = ((lower_data["iv"] + upper_data["iv"]) / 2) - middle_data["iv"]
            eligible = (
                net_credit >= SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT
                and body_iv_edge >= SHORT_CALL_BUTTERFLY_MIN_IV_EDGE
                and max_loss > 0
            )

            candidates.append({
                "reward_risk": reward_risk,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "net_credit": net_credit,
                "wing_width": wing_width,
                "body_iv_edge": body_iv_edge,
                "wing_richness": wing_richness,
                "body_distance_pct": body_distance_pct,
                "lower_symbol": lower_data["option_name"],
                "middle_symbol": middle_data["option_name"],
                "upper_symbol": upper_data["option_name"],
                "lower_strike": lower_strike,
                "middle_strike": middle_strike,
                "upper_strike": upper_strike,
                "lower_delta": lower_delta,
                "middle_delta": middle_delta,
                "upper_delta": upper_delta,
                "lower_iv": lower_data["iv"],
                "middle_iv": middle_data["iv"],
                "upper_iv": upper_data["iv"],
                "lower_bid": lower_quote.bid,
                "middle_ask": middle_quote.ask,
                "upper_bid": upper_quote.bid,
                "eligible": eligible,
            })

    return sorted(
        candidates,
        key=lambda c: (
            c["eligible"],
            c["reward_risk"],
            c["net_credit"],
            c["body_iv_edge"],
            c["wing_richness"],
            -c["body_distance_pct"],
        ),
        reverse=True,
    )


async def scan_put_spread_opportunities(asset: str = None, expiry_rank: int = PUT_SPREAD_EXPIRY_RANK):
    """
    Scan and rank all bearish put spread candidates.
    When asset is None, iterates over every symbol in ASSET_SYMBOL and appends
    a cross-asset summary table with the overall best fit at the end.
    No orders are placed.
    """
    mt5_conn = MT5Connector()
    quant_calc = QuantCalculation()
    utils = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    assets_to_scan = [asset] if asset is not None else ASSET_SYMBOL
    cross_asset_results = []

    for current_asset in assets_to_scan:
        selected = mt5_conn.symbol_select(current_asset, True)
        if not selected:
            print(f"[{current_asset}] Failed to select — skipping")
            continue

        tick_ok = False
        for _ in range(10):
            tick = mt5_conn.get_mt5_connector().symbol_info_tick(current_asset)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                tick_ok = True
                break
            print(f"Waiting for {current_asset} tick data...")
            await asyncio.sleep(1)
        if not tick_ok:
            print(f"{current_asset} has no tick data after 10 s — skipping")
            continue

        symbol_info = mt5_conn.get_symbol_info(current_asset)
        atm_price = (symbol_info.bid + symbol_info.ask) / 2

        print(f"\n{'='*72}")
        print(f"PUT SPREAD SCANNER  |  {current_asset}  |  spot={atm_price:.2f}  |  expiry rank={expiry_rank}")
        print(f"{'='*72}")

        spot_prices_data = mt5_conn.get_data(
            current_asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        print(f"GARCH vol : {garch_vol:.2f}%")

        # ── GEX: locate call wall ────────────────────────────────────────────
        mt5_raw = mt5_conn.get_mt5_connector()
        gex_result = compute_call_wall(mt5_raw, current_asset, atm_price)
        if gex_result:
            call_wall = gex_result["call_wall"]
            target_long_strike = call_wall * (1.0 + PUT_SPREAD_CALL_WALL_OFFSET)
            print(f"Call wall : {call_wall:.2f}  (OI proxy: {gex_result['oi_proxy']},  "
                  f"total call GEX={gex_result['total_call_gex']:,.0f})")
            print(f"Target long-leg strike (call_wall + {PUT_SPREAD_CALL_WALL_OFFSET*100:.0f}%) : {target_long_strike:.2f}")
            top_gex = sorted(gex_result["gex_by_strike"].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top 5 call GEX strikes: {[(f'{k:.2f}', f'{v:,.0f}') for k, v in top_gex]}")
        else:
            call_wall = None
            target_long_strike = None
            print("Call wall : unavailable (no call options with valid mid price found)")

        chain_options = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=expiry_rank
        )
        if not chain_options:
            print("No option chain returned for selected expiry rank")
            continue

        expiration_time = next(iter(chain_options.keys()))
        print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

        _, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not puts_dict:
            print("No puts returned from chain")
            continue

        # ── Full put surface ─────────────────────────────────────────────────
        print(f"{'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}  Role")
        print("-" * 60)
        for d in sorted(puts_dict, key=lambda x: abs(x), reverse=True):
            v = puts_dict[d]
            if PUT_SPREAD_LONG_DELTA_MIN <= abs(d) <= PUT_SPREAD_LONG_DELTA_MAX:
                role = "LONG  (Δ {:.2f}–{:.2f})".format(PUT_SPREAD_LONG_DELTA_MIN, PUT_SPREAD_LONG_DELTA_MAX)
            elif PUT_SPREAD_SHORT_DELTA_MIN <= abs(d) <= PUT_SPREAD_SHORT_DELTA_MAX:
                role = "SHORT (Δ {:.2f}–{:.2f})".format(PUT_SPREAD_SHORT_DELTA_MIN, PUT_SPREAD_SHORT_DELTA_MAX)
            else:
                role = ""
            print(f"{d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}  {role}")

        # ── Build all valid pairs ─────────────────────────────────────────────
        long_candidates_all = {
            d: v for d, v in puts_dict.items()
            if PUT_SPREAD_LONG_DELTA_MIN <= abs(d) <= PUT_SPREAD_LONG_DELTA_MAX
        }
        if target_long_strike is not None and long_candidates_all:
            all_long_strikes = sorted({v['strike'] for v in long_candidates_all.values()})
            below = [s for s in all_long_strikes if s <= target_long_strike]
            above = [s for s in all_long_strikes if s > target_long_strike]
            nearest_strikes = set()
            if below:
                nearest_strikes.add(max(below))
            if above:
                nearest_strikes.add(min(above))
            long_candidates = {
                d: v for d, v in long_candidates_all.items()
                if v['strike'] in nearest_strikes
            }
            print(f"\nCall-wall filter: target long strike={target_long_strike:.2f}, "
                  f"pinned to nearest available strikes={sorted(nearest_strikes)}")
            if not long_candidates:
                print("  No long candidates survive call-wall filter — falling back to full delta range.")
                long_candidates = long_candidates_all
        else:
            long_candidates = long_candidates_all

        short_candidates = {
            d: v for d, v in puts_dict.items()
            if PUT_SPREAD_SHORT_DELTA_MIN <= abs(d) <= PUT_SPREAD_SHORT_DELTA_MAX
        }

        candidates = []
        for l_delta, l_data in long_candidates.items():
            for s_delta, s_data in short_candidates.items():
                if l_data['strike'] <= s_data['strike']:
                    continue  # long leg must be the higher strike
                iv_edge = garch_vol - l_data['iv']
                spread_skew = s_data['iv'] - l_data['iv']
                proximity_penalty = 0.0
                if target_long_strike is not None:
                    proximity_penalty = abs(l_data['strike'] - target_long_strike) / max(target_long_strike, 1.0) * 10
                score = iv_edge + spread_skew - proximity_penalty
                candidates.append({
                    'score': score,
                    'iv_edge': iv_edge,
                    'spread_skew': spread_skew,
                    'proximity_penalty': proximity_penalty,
                    'long_delta': l_delta,
                    'short_delta': s_delta,
                    'long_symbol': l_data['option_name'],
                    'short_symbol': s_data['option_name'],
                    'long_iv': l_data['iv'],
                    'short_iv': s_data['iv'],
                    'long_strike': l_data['strike'],
                    'short_strike': s_data['strike'],
                })

        if not candidates:
            print(f"\n[{current_asset}] No valid spread pairs found within configured delta ranges.")
            continue

        candidates.sort(key=lambda x: x['score'], reverse=True)

        # ── Ranked table ─────────────────────────────────────────────────────
        cw_label = f"call_wall={call_wall:.2f}" if call_wall else "call_wall=N/A"
        print(f"\n{'='*80}")
        print(f"RANKED SPREAD CANDIDATES  (min iv_edge >= {PUT_SPREAD_MIN_IV_EDGE}pp  |  GARCH={garch_vol:.2f}%  |  {cw_label})")
        print(f"{'='*80}")
        col = f"{'#':>3}  {'Score':>6}  {'Edge':>6}  {'Skew':>6}  {'Prox':>6}  {'LONG leg':<18}  {'SHORT leg':<18}  {'LStr':>6}  {'SStr':>6}  {'LIV%':>6}  {'SIV%':>6}  OK"
        print(col)
        print("-" * len(col))
        for i, c in enumerate(candidates, 1):
            ok = "YES" if c['iv_edge'] >= PUT_SPREAD_MIN_IV_EDGE else "no"
            print(
                f"{i:>3}  {c['score']:>6.2f}  {c['iv_edge']:>6.2f}  {c['spread_skew']:>6.2f}  {c['proximity_penalty']:>6.2f}  "
                f"{c['long_symbol']:<18}  {c['short_symbol']:<18}  "
                f"{c['long_strike']:>6.2f}  {c['short_strike']:>6.2f}  "
                f"{c['long_iv']:>6.2f}  {c['short_iv']:>6.2f}  {ok}"
            )

        # ── Per-asset best eligible summary ──────────────────────────────────
        eligible = [c for c in candidates if c['iv_edge'] >= PUT_SPREAD_MIN_IV_EDGE]
        best = eligible[0] if eligible else candidates[0]

        cross_asset_results.append({
            "asset": current_asset,
            "garch_vol": garch_vol,
            "spot": atm_price,
            "call_wall": call_wall,
            "target_long_strike": target_long_strike,
            "best": best,
            "eligible": bool(eligible),
            "eligible_count": len(eligible),
            "total_candidates": len(candidates),
        })

        print(f"\n{'='*80}")
        if eligible:
            print(f"[{current_asset}] BEST ELIGIBLE SPREAD:")
        else:
            print(f"[{current_asset}] BEST AVAILABLE SPREAD (none met iv_edge >= {PUT_SPREAD_MIN_IV_EDGE}pp):")
        if call_wall:
            print(f"  Call wall          : {call_wall:.2f}  →  target long strike : {target_long_strike:.2f}  (+{PUT_SPREAD_CALL_WALL_OFFSET*100:.0f}%)")
        print(f"  BUY  {best['long_symbol']:<20}  strike={best['long_strike']:.2f}  Δ={best['long_delta']:.2f}  IV={best['long_iv']:.2f}%")
        print(f"  SELL {best['short_symbol']:<20}  strike={best['short_strike']:.2f}  Δ={best['short_delta']:.2f}  IV={best['short_iv']:.2f}%")
        print(f"  IV edge vs GARCH   : {best['iv_edge']:.2f}pp  |  spread skew : {best['spread_skew']:.2f}pp  |  score : {best['score']:.2f}")
        print(f"  Eligible pairs     : {len(eligible)} / {len(candidates)}")
        print(f"{'='*80}\n")

    # ── Cross-asset summary (only when scanning multiple assets) ─────────────
    if len(assets_to_scan) > 1 and cross_asset_results:
        cross_asset_results.sort(
            key=lambda r: (
                r["eligible"],
                r["best"]["score"],
                r["best"]["iv_edge"],
                r["best"]["spread_skew"],
            ),
            reverse=True,
        )
        print(f"\n{'='*110}")
        print(
            f"CROSS-ASSET PUT SPREAD SUMMARY  "
            f"({len(cross_asset_results)} of {len(assets_to_scan)} assets returned candidates  |  "
            f"min iv_edge >= {PUT_SPREAD_MIN_IV_EDGE}pp)"
        )
        print(f"{'='*110}")
        col = (
            f"{'#':>3}  {'Asset':<8}  {'GARCH%':>7}  {'Spot':>8}  {'Score':>6}  {'Edge':>6}  "
            f"{'Skew':>6}  {'Elig':>4}  {'LONG leg':<18}  {'SHORT leg':<18}  {'LStr':>6}  {'SStr':>6}  OK"
        )
        print(col)
        print("-" * len(col))
        for i, r in enumerate(cross_asset_results, 1):
            b = r["best"]
            ok = "YES" if r["eligible"] else "no"
            print(
                f"{i:>3}  {r['asset']:<8}  {r['garch_vol']:>7.2f}  {r['spot']:>8.2f}  "
                f"{b['score']:>6.2f}  {b['iv_edge']:>6.2f}  {b['spread_skew']:>6.2f}  "
                f"{r['eligible_count']:>4}  "
                f"{b['long_symbol']:<18}  {b['short_symbol']:<18}  "
                f"{b['long_strike']:>6.2f}  {b['short_strike']:>6.2f}  {ok}"
            )

        overall = cross_asset_results[0]
        ob = overall["best"]
        print(f"\n{'='*110}")
        status = "ELIGIBLE" if overall["eligible"] else "AVAILABLE (below iv_edge threshold)"
        print(
            f"OVERALL BEST {status}:  {overall['asset']}  |  "
            f"GARCH={overall['garch_vol']:.2f}%  |  spot={overall['spot']:.2f}"
        )
        if overall["call_wall"]:
            print(
                f"  Call wall          : {overall['call_wall']:.2f}  →  "
                f"target long strike : {overall['target_long_strike']:.2f}  (+{PUT_SPREAD_CALL_WALL_OFFSET*100:.0f}%)"
            )
        print(f"  BUY  {ob['long_symbol']:<20}  strike={ob['long_strike']:.2f}  Δ={ob['long_delta']:.2f}  IV={ob['long_iv']:.2f}%")
        print(f"  SELL {ob['short_symbol']:<20}  strike={ob['short_strike']:.2f}  Δ={ob['short_delta']:.2f}  IV={ob['short_iv']:.2f}%")
        print(
            f"  IV edge vs GARCH   : {ob['iv_edge']:.2f}pp  |  "
            f"spread skew : {ob['spread_skew']:.2f}pp  |  score : {ob['score']:.2f}"
        )
        print(f"{'='*110}\n")


async def scan_short_call_butterfly_opportunities(
    asset: str = None,
    expiry_rank: int = SHORT_CALL_BUTTERFLY_EXPIRY_RANK,
):
    """
    Scan and rank all symmetric short call butterfly candidates.
    When asset is None, iterates over every symbol in ASSET_SYMBOL and appends
    a cross-asset summary table with the overall best fit at the end.
    No orders are placed.
    """
    mt5_conn = MT5Connector()
    quant_calc = QuantCalculation()
    utils = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    assets_to_scan = [asset] if asset is not None else ASSET_SYMBOL
    cross_asset_results = []

    for current_asset in assets_to_scan:
        selected = mt5_conn.symbol_select(current_asset, True)
        if not selected:
            print(f"[{current_asset}] Failed to select — skipping")
            continue

        tick_ok = False
        for _ in range(10):
            tick = mt5_conn.get_mt5_connector().symbol_info_tick(current_asset)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                tick_ok = True
                break
            print(f"Waiting for {current_asset} tick data...")
            await asyncio.sleep(1)
        if not tick_ok:
            print(f"{current_asset} has no tick data after 10 s — skipping")
            continue

        symbol_info = mt5_conn.get_symbol_info(current_asset)
        spot = (symbol_info.bid + symbol_info.ask) / 2

        print(f"\n{'='*88}")
        print(f"SHORT CALL BUTTERFLY SCANNER  |  {current_asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
        print(f"{'='*88}")

        spot_prices_data = mt5_conn.get_data(
            current_asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        print(f"GARCH vol : {garch_vol:.2f}%")

        chain_options = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=expiry_rank
        )
        if not chain_options:
            print("No option chain returned for selected expiry rank")
            continue

        expiration_time = next(iter(chain_options.keys()))
        print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

        calls_dict, _ = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not calls_dict:
            print("No calls returned from chain")
            continue

        print(f"{'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}")
        print("-" * 52)
        for d in sorted(calls_dict):
            v = calls_dict[d]
            print(f"{d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}")

        candidates = build_short_call_butterfly_candidates(utils, calls_dict, spot, garch_vol)
        if not candidates:
            print("\nNo symmetric short call butterfly candidates found with valid live quotes.")
            continue

        print(f"\n{'='*136}")
        print(
            "RANKED SHORT CALL BUTTERFLY CANDIDATES  "
            f"(min body_iv_edge >= {SHORT_CALL_BUTTERFLY_MIN_IV_EDGE:.2f}pp  |  "
            f"min credit >= {SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT:.4f}  |  "
            f"max body distance <= {SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT*100:.1f}%  |  "
            f"GARCH={garch_vol:.2f}%)"
        )
        print(f"{'='*136}")
        col = (
            f"{'#':>3}  {'RR':>7}  {'Credit':>8}  {'MaxLoss':>8}  {'Wing':>6}  "
            f"{'Edge':>6}  {'Rich':>6}  {'Dist%':>6}  {'LOW':<16}  {'BODY':<16}  {'UP':<16}  OK"
        )
        print(col)
        print("-" * len(col))
        for i, c in enumerate(candidates, 1):
            rr_display = f"{c['reward_risk']:.2f}" if math.isfinite(c["reward_risk"]) else "N/A"
            ok = "YES" if c["eligible"] else "no"
            print(
                f"{i:>3}  {rr_display:>7}  {c['net_credit']:>8.4f}  {c['max_loss']:>8.4f}  {c['wing_width']:>6.2f}  "
                f"{c['body_iv_edge']:>6.2f}  {c['wing_richness']:>6.2f}  {c['body_distance_pct']*100:>6.2f}  "
                f"{c['lower_symbol']:<16}  {c['middle_symbol']:<16}  {c['upper_symbol']:<16}  {ok}"
            )

        eligible = [c for c in candidates if c["eligible"]]
        best = eligible[0] if eligible else candidates[0]

        cross_asset_results.append({
            "asset": current_asset,
            "garch_vol": garch_vol,
            "spot": spot,
            "best": best,
            "eligible": bool(eligible),
        })

        print(f"\n{'='*88}")
        if eligible:
            print(f"[{current_asset}] BEST ELIGIBLE SHORT CALL BUTTERFLY:")
        else:
            print(f"[{current_asset}] BEST AVAILABLE SHORT CALL BUTTERFLY (none met all thresholds):")
        print(
            f"  SELL {best['lower_symbol']:<20} strike={best['lower_strike']:.2f}  Δ={best['lower_delta']:.2f}  "
            f"IV={best['lower_iv']:.2f}%  bid={best['lower_bid']:.4f}"
        )
        print(
            f"  BUY  2x {best['middle_symbol']:<17} strike={best['middle_strike']:.2f}  Δ={best['middle_delta']:.2f}  "
            f"IV={best['middle_iv']:.2f}%  ask={best['middle_ask']:.4f}"
        )
        print(
            f"  SELL {best['upper_symbol']:<20} strike={best['upper_strike']:.2f}  Δ={best['upper_delta']:.2f}  "
            f"IV={best['upper_iv']:.2f}%  bid={best['upper_bid']:.4f}"
        )
        rr_summary = f"{best['reward_risk']:.2f}" if math.isfinite(best["reward_risk"]) else "N/A"
        print(
            f"  Net credit         : {best['net_credit']:.4f}  |  Max profit : {best['max_profit']:.4f}  "
            f"|  Max loss : {best['max_loss']:.4f}  |  Reward/Risk : {rr_summary}"
        )
        print(
            f"  Body IV edge       : {best['body_iv_edge']:.2f}pp vs GARCH  |  "
            f"Wing richness : {best['wing_richness']:.2f}pp  |  "
            f"Body distance : {best['body_distance_pct']*100:.2f}% from spot"
        )
        print(f"{'='*88}\n")

    # ── Cross-asset summary (only when scanning multiple assets) ─────────────
    if len(assets_to_scan) > 1 and cross_asset_results:
        cross_asset_results.sort(
            key=lambda r: (
                r["eligible"],
                r["best"]["reward_risk"] if math.isfinite(r["best"]["reward_risk"]) else -math.inf,
                r["best"]["net_credit"],
                r["best"]["body_iv_edge"],
            ),
            reverse=True,
        )
        print(f"\n{'='*120}")
        print(f"CROSS-ASSET SHORT CALL BUTTERFLY SUMMARY  ({len(cross_asset_results)} of {len(assets_to_scan)} assets returned candidates)")
        print(f"{'='*120}")
        col = (
            f"{'#':>3}  {'Asset':<8}  {'RR':>7}  {'Credit':>8}  {'MaxLoss':>8}  {'Edge':>6}  "
            f"{'Rich':>6}  {'Dist%':>6}  {'GARCHvol%':>10}  {'BODY':<20}  OK"
        )
        print(col)
        print("-" * len(col))
        for i, r in enumerate(cross_asset_results, 1):
            b = r["best"]
            rr_display = f"{b['reward_risk']:.2f}" if math.isfinite(b["reward_risk"]) else "N/A"
            ok = "YES" if r["eligible"] else "no"
            print(
                f"{i:>3}  {r['asset']:<8}  {rr_display:>7}  {b['net_credit']:>8.4f}  {b['max_loss']:>8.4f}  "
                f"{b['body_iv_edge']:>6.2f}  {b['wing_richness']:>6.2f}  {b['body_distance_pct']*100:>6.2f}  "
                f"{r['garch_vol']:>10.2f}  {b['middle_symbol']:<20}  {ok}"
            )

        overall = cross_asset_results[0]
        ob = overall["best"]
        print(f"\n{'='*120}")
        status = "ELIGIBLE" if overall["eligible"] else "AVAILABLE (below thresholds)"
        print(
            f"OVERALL BEST {status}:  {overall['asset']}  |  "
            f"GARCH={overall['garch_vol']:.2f}%  |  spot={overall['spot']:.2f}"
        )
        print(
            f"  SELL {ob['lower_symbol']:<20} strike={ob['lower_strike']:.2f}  Δ={ob['lower_delta']:.2f}  "
            f"IV={ob['lower_iv']:.2f}%  bid={ob['lower_bid']:.4f}"
        )
        print(
            f"  BUY  2x {ob['middle_symbol']:<17} strike={ob['middle_strike']:.2f}  Δ={ob['middle_delta']:.2f}  "
            f"IV={ob['middle_iv']:.2f}%  ask={ob['middle_ask']:.4f}"
        )
        print(
            f"  SELL {ob['upper_symbol']:<20} strike={ob['upper_strike']:.2f}  Δ={ob['upper_delta']:.2f}  "
            f"IV={ob['upper_iv']:.2f}%  bid={ob['upper_bid']:.4f}"
        )
        rr_summary = f"{ob['reward_risk']:.2f}" if math.isfinite(ob["reward_risk"]) else "N/A"
        print(
            f"  Net credit         : {ob['net_credit']:.4f}  |  Max profit : {ob['max_profit']:.4f}  "
            f"|  Max loss : {ob['max_loss']:.4f}  |  Reward/Risk : {rr_summary}"
        )
        print(
            f"  Body IV edge       : {ob['body_iv_edge']:.2f}pp vs GARCH  |  "
            f"Wing richness : {ob['wing_richness']:.2f}pp  |  "
            f"Body distance : {ob['body_distance_pct']*100:.2f}% from spot"
        )
        print(f"{'='*120}\n")


async def scan_straddle_opportunities(
    asset: str = None,
    expiry_rank: int = TARGET_OPTION_EXPIRY_RANK,
):
    """
    Scan and rank ATM straddle candidates (same-strike call + put) for the given asset.
    Ranks by IV edge vs GARCH (long straddle: cheap IV) filtered by delta neutrality.
    No orders are placed.
    """
    mt5_conn = MT5Connector()
    quant_calc = QuantCalculation()
    utils = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    if asset is None:
        asset = ASSET_SYMBOL[0]

    selected = mt5_conn.symbol_select(asset, True)
    if not selected:
        print(f"Failed to select {asset}")
        return

    for _ in range(10):
        tick = mt5_conn.get_mt5_connector().symbol_info_tick(asset)
        if tick is not None and tick.bid > 0 and tick.ask > 0:
            break
        print(f"Waiting for {asset} tick data...")
        await asyncio.sleep(1)
    else:
        print(f"{asset} has no tick data after 10 s — check Market Watch")
        return

    symbol_info = mt5_conn.get_symbol_info(asset)
    spot = (symbol_info.bid + symbol_info.ask) / 2

    print(f"\n{'='*80}")
    print(f"STRADDLE SCANNER  |  {asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
    print(f"{'='*80}")

    spot_prices_data = mt5_conn.get_data(
        asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
    )["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
    print(f"GARCH vol : {garch_vol:.2f}%")

    chain_options = mt5_conn.get_option_names_by_expiration_time(
        asset, expiry_rank_override=expiry_rank
    )
    if not chain_options:
        print("No option chain returned for selected expiry rank")
        return

    expiration_time = next(iter(chain_options.keys()))
    print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

    calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
    if not calls_dict or not puts_dict:
        print("No calls or puts returned from chain")
        return

    # Build strike maps
    call_by_strike = {}
    for call_delta, call_data in calls_dict.items():
        strike = call_data["strike"]
        current = call_by_strike.get(strike)
        if current is None or abs(call_delta - 0.50) < abs(current[0] - 0.50):
            call_by_strike[strike] = (call_delta, call_data)

    put_by_strike = {}
    for put_delta, put_data in puts_dict.items():
        strike = put_data["strike"]
        current = put_by_strike.get(strike)
        if current is None or abs(abs(put_delta) - 0.50) < abs(abs(current[0]) - 0.50):
            put_by_strike[strike] = (put_delta, put_data)

    common_strikes = sorted(set(call_by_strike.keys()) & set(put_by_strike.keys()))
    if not common_strikes:
        print("No common strikes between calls and puts")
        return

    # Build candidates for all common strikes
    candidates = []
    for strike in common_strikes:
        call_delta, call_data = call_by_strike[strike]
        put_delta, put_data = put_by_strike[strike]

        call_name = call_data["option_name"]
        put_name = put_data["option_name"]

        if not mt5_conn.symbol_select(call_name, True):
            print(f"  [skip] Could not select {call_name}")
            continue
        if not mt5_conn.symbol_select(put_name, True):
            print(f"  [skip] Could not select {put_name}")
            continue

        call_quote = utils.get_option_info_with_quote(call_name)
        put_quote = utils.get_option_info_with_quote(put_name)
        if (call_quote is None or call_quote.bid <= 0 or call_quote.ask <= 0 or
                put_quote is None or put_quote.bid <= 0 or put_quote.ask <= 0):
            continue

        net_iv = (call_data["iv"] + put_data["iv"]) / 2
        iv_edge = garch_vol - net_iv          # positive = IV cheap vs GARCH (good for long straddle)
        delta_imbalance = abs(call_delta + put_delta)  # 0 = perfectly neutral
        straddle_cost = call_quote.ask + put_quote.ask
        distance_pct = abs(strike - spot) / max(spot, 1.0)

        eligible = (
            net_iv <= STRADDLE_ENTRY_MAX_NET_IV
            and delta_imbalance <= STRADDLE_MAX_DELTA_IMBALANCE
            and iv_edge > 0
        )

        candidates.append({
            "strike": strike,
            "call_symbol": call_data["option_name"],
            "put_symbol": put_data["option_name"],
            "call_delta": call_delta,
            "put_delta": put_delta,
            "call_iv": call_data["iv"],
            "put_iv": put_data["iv"],
            "net_iv": net_iv,
            "iv_edge": iv_edge,
            "delta_imbalance": delta_imbalance,
            "straddle_cost": straddle_cost,
            "call_ask": call_quote.ask,
            "put_ask": put_quote.ask,
            "distance_pct": distance_pct,
            "eligible": eligible,
        })

    if not candidates:
        print("No straddle candidates found with valid live quotes.")
        return

    # Sort: eligible first, then by iv_edge desc, then delta_imbalance asc
    candidates.sort(
        key=lambda c: (c["eligible"], c["iv_edge"], -c["delta_imbalance"]),
        reverse=True,
    )

    # ── Full surface table ────────────────────────────────────────────────────
    print(f"{'Strike':>8}  {'CallΔ':>7}  {'PutΔ':>7}  {'|ΔNet|':>7}  {'CallIV%':>8}  {'PutIV%':>8}  {'NetIV%':>8}  {'Edge':>6}  {'Cost':>8}  {'Dist%':>6}  OK")
    print("-" * 100)
    for c in candidates:
        ok = "YES" if c["eligible"] else "no"
        print(
            f"{c['strike']:>8.2f}  {c['call_delta']:>7.2f}  {c['put_delta']:>7.2f}  "
            f"{c['delta_imbalance']:>7.4f}  {c['call_iv']:>8.2f}  {c['put_iv']:>8.2f}  "
            f"{c['net_iv']:>8.2f}  {c['iv_edge']:>6.2f}  {c['straddle_cost']:>8.4f}  "
            f"{c['distance_pct']*100:>6.2f}  {ok}"
        )

    # ── Best eligible summary ─────────────────────────────────────────────────
    eligible = [c for c in candidates if c["eligible"]]
    print(f"\n{'='*80}")
    if eligible:
        b = eligible[0]
        print("BEST ELIGIBLE STRADDLE:")
        print(f"  BUY  {b['call_symbol']:<20}  strike={b['strike']:.2f}  Δ={b['call_delta']:.2f}  IV={b['call_iv']:.2f}%  ask={b['call_ask']:.4f}")
        print(f"  BUY  {b['put_symbol']:<20}  strike={b['strike']:.2f}  Δ={b['put_delta']:.2f}  IV={b['put_iv']:.2f}%  ask={b['put_ask']:.4f}")
        print(f"  Net IV         : {b['net_iv']:.2f}%  |  IV edge vs GARCH : {b['iv_edge']:.2f}pp  |  |ΔNet| : {b['delta_imbalance']:.4f}")
        print(f"  Total cost     : {b['straddle_cost']:.4f}  |  Distance from spot : {b['distance_pct']*100:.2f}%")
    else:
        print(
            f"No straddle meets thresholds: net_iv <= {STRADDLE_ENTRY_MAX_NET_IV}%  AND  "
            f"|ΔNet| <= {STRADDLE_MAX_DELTA_IMBALANCE}  AND  iv_edge > 0  (GARCH={garch_vol:.2f}%)"
        )

    # ── Best overall by RR and by IV (unconditional — across all candidates) ──
    # RR proxy = IV edge per unit of premium paid (more edge per dollar = better RR).
    # Best IV  = lowest net_iv (cheapest straddle in implied vol terms).
    best_rr  = max(candidates, key=lambda c: c["iv_edge"] / max(c["straddle_cost"], 1e-6))
    best_iv  = min(candidates, key=lambda c: c["net_iv"])

    def _straddle_line(label: str, c: dict) -> None:
        rr_ratio = c["iv_edge"] / max(c["straddle_cost"], 1e-6)
        print(f"  [{label}]  strike={c['strike']:.2f}  Δ_net={c['call_delta']+c['put_delta']:+.4f}")
        print(f"    BUY  {c['call_symbol']:<20}  Δ={c['call_delta']:+.4f}  IV={c['call_iv']:.2f}%  ask={c['call_ask']:.4f}")
        print(f"    BUY  {c['put_symbol']:<20}  Δ={c['put_delta']:+.4f}  IV={c['put_iv']:.2f}%  ask={c['put_ask']:.4f}")
        print(f"    Net IV={c['net_iv']:.2f}%  IV_edge={c['iv_edge']:+.2f}pp  cost={c['straddle_cost']:.4f}  RR={rr_ratio:.4f}  dist={c['distance_pct']*100:.2f}%")

    print(f"\n{'─'*80}")
    print(f"  BEST OVERALL OPPORTUNITIES  (GARCH={garch_vol:.2f}%  |  all {len(candidates)} candidates)")
    print(f"{'─'*80}")
    _straddle_line("BEST RR  — max IV-edge/cost", best_rr)
    print()
    _straddle_line("BEST IV  — lowest net IV   ", best_iv)
    print(f"{'='*80}\n")


async def scan_all_straddles(expiry_rank: int = TARGET_OPTION_EXPIRY_RANK):
    for asset in ASSET_SYMBOL:
        await scan_straddle_opportunities(asset, expiry_rank=expiry_rank)


# ══════════════════════════════════════════════════════════════════════════════
# SHORT STRANGLE SCANNER
# For each asset, find the best OTM call + OTM put pair to sell as a strangle.
# Scoring rewards: high IV edge on both legs (IV > GARCH), high net credit,
# and near delta-neutrality.  Filtering enforces min IV edge per leg and
# a max delta imbalance.
# ══════════════════════════════════════════════════════════════════════════════

async def scan_short_strangle_opportunities(
    asset: str,
    expiry_rank: int = SHORT_STRANGLE_EXPIRY_RANK,
):
    mt5_conn = MT5Connector()
    utils = Utils()
    quant_calc = QuantCalculation()

    if not mt5_conn.initialize():
        print(f"[{asset}] MT5 initialization failed")
        return

    if not mt5_conn.symbol_select(asset, True):
        print(f"[{asset}] Failed to select underlying")
        return

    for _ in range(10):
        tick = mt5_conn.get_mt5_connector().symbol_info_tick(asset)
        if tick is not None and tick.bid > 0 and tick.ask > 0:
            break
        print(f"[{asset}] Waiting for tick data...")
        await asyncio.sleep(1)
    else:
        print(f"[{asset}] No tick data after 10 s — skipping")
        return

    symbol_info = mt5_conn.get_symbol_info(asset)
    spot = (symbol_info.bid + symbol_info.ask) / 2

    print(f"\n{'='*80}")
    print(f"SHORT STRANGLE SCANNER  |  {asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
    print(f"{'='*80}")

    spot_prices_data = mt5_conn.get_data(
        asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
    )["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
    print(f"GARCH vol : {garch_vol:.2f}%")

    chain_options = mt5_conn.get_option_names_by_expiration_time(
        asset, expiry_rank_override=expiry_rank
    )
    if not chain_options:
        print(f"[{asset}] No option chain for expiry rank {expiry_rank}")
        return

    expiration_time = next(iter(chain_options.keys()))
    print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

    calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
    if not calls_dict or not puts_dict:
        print(f"[{asset}] No calls or puts returned from chain")
        return

    # ── Filter legs by delta range ────────────────────────────────────────────
    # OTM calls: positive delta between [CALL_DELTA_MIN, CALL_DELTA_MAX]
    otm_calls = {
        delta: data
        for delta, data in calls_dict.items()
        if SHORT_STRANGLE_CALL_DELTA_MIN <= delta <= SHORT_STRANGLE_CALL_DELTA_MAX
    }
    # OTM puts: negative delta; filter on abs(delta) in [PUT_DELTA_MIN, PUT_DELTA_MAX]
    otm_puts = {
        delta: data
        for delta, data in puts_dict.items()
        if SHORT_STRANGLE_PUT_DELTA_MIN <= abs(delta) <= SHORT_STRANGLE_PUT_DELTA_MAX
    }

    if not otm_calls:
        print(f"[{asset}] No OTM calls within delta range "
              f"[{SHORT_STRANGLE_CALL_DELTA_MIN}, {SHORT_STRANGLE_CALL_DELTA_MAX}]")
        return
    if not otm_puts:
        print(f"[{asset}] No OTM puts within delta range "
              f"[{SHORT_STRANGLE_PUT_DELTA_MIN}, {SHORT_STRANGLE_PUT_DELTA_MAX}]")
        return

    # ── Build candidates ──────────────────────────────────────────────────────
    candidates = []
    for call_delta, call_data in otm_calls.items():
        call_name = call_data["option_name"]
        if not mt5_conn.symbol_select(call_name, True):
            continue
        call_quote = utils.get_option_info_with_quote(call_name)
        if call_quote is None or call_quote.bid <= 0 or call_quote.ask <= 0:
            continue

        for put_delta, put_data in otm_puts.items():
            # Strangle: call strike must be above put strike
            if call_data["strike"] <= put_data["strike"]:
                continue

            put_name = put_data["option_name"]
            if not mt5_conn.symbol_select(put_name, True):
                continue
            put_quote = utils.get_option_info_with_quote(put_name)
            if put_quote is None or put_quote.bid <= 0 or put_quote.ask <= 0:
                continue

            net_credit      = call_quote.bid + put_quote.bid
            call_iv_edge    = call_data["iv"] - garch_vol
            put_iv_edge     = put_data["iv"]  - garch_vol
            delta_imbalance = abs(call_delta + put_delta)
            call_dist_pct   = (call_data["strike"] - spot) / max(spot, 1.0)
            put_dist_pct    = (spot - put_data["strike"])  / max(spot, 1.0)
            width_pct       = (call_data["strike"] - put_data["strike"]) / max(spot, 1.0)

            eligible = (
                call_iv_edge    >= SHORT_STRANGLE_MIN_IV_EDGE
                and put_iv_edge >= SHORT_STRANGLE_MIN_IV_EDGE
                and delta_imbalance <= SHORT_STRANGLE_MAX_DELTA_IMBALANCE
                and net_credit  >= SHORT_STRANGLE_MIN_NET_CREDIT
            )

            # Score: rewards high IV edge on both legs, high premium, low delta bias
            score = (
                call_iv_edge
                + put_iv_edge
                + net_credit
                - delta_imbalance * 5
            )

            candidates.append({
                "score":           score,
                "eligible":        eligible,
                "call_symbol":     call_name,
                "put_symbol":      put_name,
                "call_strike":     call_data["strike"],
                "put_strike":      put_data["strike"],
                "call_delta":      call_delta,
                "put_delta":       put_delta,
                "call_iv":         call_data["iv"],
                "put_iv":          put_data["iv"],
                "call_iv_edge":    call_iv_edge,
                "put_iv_edge":     put_iv_edge,
                "delta_imbalance": delta_imbalance,
                "net_credit":      net_credit,
                "call_bid":        call_quote.bid,
                "put_bid":         put_quote.bid,
                "call_dist_pct":   call_dist_pct,
                "put_dist_pct":    put_dist_pct,
                "width_pct":       width_pct,
                "garch_vol":       garch_vol,
            })

    if not candidates:
        print(f"[{asset}] No valid strangle candidates found with live quotes.")
        return

    candidates.sort(key=lambda c: (c["eligible"], c["score"]), reverse=True)

    # ── Full table ────────────────────────────────────────────────────────────
    header = (
        f"{'CallK':>7}  {'CΔ':>6}  {'CIV%':>7}  {'CEdge':>6}  "
        f"{'PutK':>7}  {'PΔ':>6}  {'PIV%':>7}  {'PEdge':>6}  "
        f"{'|ΔNet|':>6}  {'Credit':>7}  {'Width%':>6}  OK"
    )
    print(header)
    print("-" * len(header))
    for c in candidates[:20]:  # cap table at 20 rows
        ok = "YES" if c["eligible"] else "no"
        print(
            f"{c['call_strike']:>7.2f}  {c['call_delta']:>6.2f}  {c['call_iv']:>7.2f}  {c['call_iv_edge']:>6.2f}  "
            f"{c['put_strike']:>7.2f}  {c['put_delta']:>6.2f}  {c['put_iv']:>7.2f}  {c['put_iv_edge']:>6.2f}  "
            f"{c['delta_imbalance']:>6.4f}  {c['net_credit']:>7.4f}  {c['width_pct']*100:>6.2f}  {ok}"
        )

    # ── Best eligible summary ─────────────────────────────────────────────────
    eligible_list = [c for c in candidates if c["eligible"]]
    print(f"\n{'='*80}")
    if eligible_list:
        b = eligible_list[0]
        print("BEST SHORT STRANGLE:")
        print(f"  SELL {b['call_symbol']:<20}  strike={b['call_strike']:.2f}  Δ={b['call_delta']:.2f}  IV={b['call_iv']:.2f}%  edge={b['call_iv_edge']:+.2f}pp  bid={b['call_bid']:.4f}")
        print(f"  SELL {b['put_symbol']:<20}  strike={b['put_strike']:.2f}  Δ={b['put_delta']:.2f}  IV={b['put_iv']:.2f}%  edge={b['put_iv_edge']:+.2f}pp  bid={b['put_bid']:.4f}")
        print(f"  Net credit     : {b['net_credit']:.4f}")
        print(f"  |ΔNet|         : {b['delta_imbalance']:.4f}")
        print(f"  Width          : {b['width_pct']*100:.2f}%  (call {b['call_dist_pct']*100:.2f}% above / put {b['put_dist_pct']*100:.2f}% below spot)")
        print(f"  GARCH vol      : {b['garch_vol']:.2f}%")
        print(f"  Score          : {b['score']:.4f}")
    else:
        print(
            f"No strangle meets thresholds: "
            f"call/put IV edge >= {SHORT_STRANGLE_MIN_IV_EDGE}pp  AND  "
            f"|ΔNet| <= {SHORT_STRANGLE_MAX_DELTA_IMBALANCE}  AND  "
            f"credit >= {SHORT_STRANGLE_MIN_NET_CREDIT}  (GARCH={garch_vol:.2f}%)"
        )
    print(f"{'='*80}\n")

    return eligible_list[0] if eligible_list else None


async def scan_all_short_strangles(expiry_rank: int = SHORT_STRANGLE_EXPIRY_RANK):
    """Scan every asset in ASSET_SYMBOL and rank the best short strangle across all."""
    results = []
    for asset in ASSET_SYMBOL:
        best = await scan_short_strangle_opportunities(asset, expiry_rank=expiry_rank)
        if best is not None:
            results.append((asset, best))

    if not results:
        print("\nNo eligible short strangle found across all assets.")
        return

    results.sort(key=lambda x: x[1]["score"], reverse=True)

    print("\n" + "=" * 80)
    print("CROSS-ASSET SHORT STRANGLE RANKING")
    print("=" * 80)
    print(f"{'Rank':>4}  {'Asset':<8}  {'CallK':>7}  {'PutK':>7}  {'Credit':>7}  {'CEdge':>6}  {'PEdge':>6}  {'Score':>7}")
    print("-" * 70)
    for rank, (asset, c) in enumerate(results, 1):
        print(
            f"{rank:>4}  {asset:<8}  {c['call_strike']:>7.2f}  {c['put_strike']:>7.2f}  "
            f"{c['net_credit']:>7.4f}  {c['call_iv_edge']:>6.2f}  {c['put_iv_edge']:>6.2f}  {c['score']:>7.4f}"
        )
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# LONG CALL BUTTERFLY SCANNER
# Structure: BUY lower call  +  SELL 2x middle call  +  BUY upper call
# Net debit.  Max profit if spot finishes at the body strike at expiry.
# Enter when body IV is rich vs GARCH (selling overpriced vol) and wings are
# cheap vs GARCH (buying underpriced vol).
# ══════════════════════════════════════════════════════════════════════════════

def build_long_call_butterfly_candidates(utils, calls_dict, spot_price, garch_vol):
    """
    Build and rank all symmetric long call butterfly candidates.
    Symmetric means: lower_strike = middle - wing_width, upper_strike = middle + wing_width.

    Scoring rewards:
      - body_richness  : middle_iv - garch_vol  (positive → body is overpriced → good to sell)
      - wing_cheapness : garch_vol - avg(lower_iv, upper_iv)  (positive → wings cheap → good to buy)
      - reward_risk    : max_profit / net_debit
    """
    # De-duplicate by strike, keeping entry closest to delta 0.50
    call_by_strike = {}
    for call_delta, call_data in calls_dict.items():
        strike = call_data["strike"]
        current = call_by_strike.get(strike)
        if current is None or abs(call_delta - 0.50) < abs(current[0] - 0.50):
            call_by_strike[strike] = (call_delta, call_data)

    if len(call_by_strike) < 3:
        return []

    # Fetch live quotes once per strike
    quote_by_strike = {}
    for strike, (_, call_data) in call_by_strike.items():
        option_info = utils.get_option_info_with_quote(call_data["option_name"])
        if option_info is None or option_info.bid <= 0.0 or option_info.ask <= 0.0:
            continue
        quote_by_strike[strike] = option_info

    strikes = sorted(quote_by_strike.keys())
    strikes_set = set(strikes)
    candidates = []

    for middle_idx in range(1, len(strikes) - 1):
        middle_strike = strikes[middle_idx]
        body_distance_pct = abs(middle_strike - spot_price) / max(spot_price, 1.0)
        if body_distance_pct > LONG_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT:
            continue

        for lower_strike in strikes[:middle_idx]:
            wing_width = middle_strike - lower_strike
            if wing_width <= 0:
                continue

            upper_strike = round(middle_strike + wing_width, 8)
            if upper_strike not in strikes_set:
                continue

            lower_delta, lower_data = call_by_strike[lower_strike]
            middle_delta, middle_data = call_by_strike[middle_strike]
            upper_delta, upper_data = call_by_strike[upper_strike]
            lower_quote = quote_by_strike[lower_strike]
            middle_quote = quote_by_strike[middle_strike]
            upper_quote = quote_by_strike[upper_strike]

            # Long butterfly: BUY lower (ask) + SELL 2x middle (bid) + BUY upper (ask)
            net_debit    = lower_quote.ask + upper_quote.ask - (2 * middle_quote.bid)
            max_profit   = wing_width - net_debit
            max_loss     = net_debit  # limited to premium paid
            reward_risk  = max_profit / net_debit if net_debit > 0 and max_profit > 0 else -math.inf

            body_richness  = middle_data["iv"] - garch_vol        # >0: body overpriced vs GARCH
            wing_cheapness = garch_vol - (lower_data["iv"] + upper_data["iv"]) / 2  # >0: wings cheap vs GARCH

            eligible = (
                net_debit > 0
                and max_profit > 0
                and body_richness  >= LONG_CALL_BUTTERFLY_MIN_BODY_RICHNESS
                and reward_risk    >= LONG_CALL_BUTTERFLY_MIN_REWARD_RISK
            )

            score = body_richness + wing_cheapness + reward_risk - body_distance_pct * 100

            candidates.append({
                "score":              score,
                "reward_risk":        reward_risk,
                "max_profit":         max_profit,
                "max_loss":           max_loss,
                "net_debit":          net_debit,
                "wing_width":         wing_width,
                "body_richness":      body_richness,
                "wing_cheapness":     wing_cheapness,
                "body_distance_pct":  body_distance_pct,
                "lower_symbol":       lower_data["option_name"],
                "middle_symbol":      middle_data["option_name"],
                "upper_symbol":       upper_data["option_name"],
                "lower_strike":       lower_strike,
                "middle_strike":      middle_strike,
                "upper_strike":       upper_strike,
                "lower_delta":        lower_delta,
                "middle_delta":       middle_delta,
                "upper_delta":        upper_delta,
                "lower_iv":           lower_data["iv"],
                "middle_iv":          middle_data["iv"],
                "upper_iv":           upper_data["iv"],
                "lower_ask":          lower_quote.ask,
                "middle_bid":         middle_quote.bid,
                "upper_ask":          upper_quote.ask,
                "eligible":           eligible,
            })

    return sorted(
        candidates,
        key=lambda c: (c["eligible"], c["score"]),
        reverse=True,
    )


async def scan_long_call_butterfly_opportunities(
    asset: str = None,
    expiry_rank: int = LONG_CALL_BUTTERFLY_EXPIRY_RANK,
):
    """
    Scan and rank all symmetric long call butterfly candidates.
    When asset is None, iterates over every symbol in ASSET_SYMBOL and prints
    a cross-asset summary at the end.  No orders are placed.
    """
    mt5_conn = MT5Connector()
    quant_calc = QuantCalculation()
    utils = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    assets_to_scan = [asset] if asset is not None else ASSET_SYMBOL
    cross_asset_results = []

    for current_asset in assets_to_scan:
        if not mt5_conn.symbol_select(current_asset, True):
            print(f"[{current_asset}] Failed to select — skipping")
            continue

        tick_ok = False
        for _ in range(10):
            tick = mt5_conn.get_mt5_connector().symbol_info_tick(current_asset)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                tick_ok = True
                break
            print(f"Waiting for {current_asset} tick data...")
            await asyncio.sleep(1)
        if not tick_ok:
            print(f"{current_asset} has no tick data after 10 s — skipping")
            continue

        symbol_info = mt5_conn.get_symbol_info(current_asset)
        spot = (symbol_info.bid + symbol_info.ask) / 2

        print(f"\n{'='*88}")
        print(f"LONG CALL BUTTERFLY SCANNER  |  {current_asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
        print(f"{'='*88}")

        spot_prices_data = mt5_conn.get_data(
            current_asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        print(f"GARCH vol : {garch_vol:.2f}%")

        chain_options = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=expiry_rank
        )
        if not chain_options:
            print("No option chain returned for selected expiry rank")
            continue

        expiration_time = next(iter(chain_options.keys()))
        print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

        calls_dict, _ = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not calls_dict:
            print("No calls returned from chain")
            continue

        print(f"{'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}")
        print("-" * 52)
        for d in sorted(calls_dict):
            v = calls_dict[d]
            print(f"{d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}")

        candidates = build_long_call_butterfly_candidates(utils, calls_dict, spot, garch_vol)
        if not candidates:
            print("\nNo symmetric long call butterfly candidates found with valid live quotes.")
            continue

        print(f"\n{'='*136}")
        print(
            "RANKED LONG CALL BUTTERFLY CANDIDATES  "
            f"(min body_richness >= {LONG_CALL_BUTTERFLY_MIN_BODY_RICHNESS:.2f}pp  |  "
            f"min R/R >= {LONG_CALL_BUTTERFLY_MIN_REWARD_RISK:.2f}  |  "
            f"max body distance <= {LONG_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT*100:.1f}%  |  "
            f"GARCH={garch_vol:.2f}%)"
        )
        print(f"{'='*136}")
        col = (
            f"{'#':>3}  {'RR':>7}  {'Debit':>8}  {'MaxProfit':>9}  {'Wing':>6}  "
            f"{'BdRich':>7}  {'WgCheap':>8}  {'Dist%':>6}  {'LOW':<16}  {'BODY':<16}  {'UP':<16}  OK"
        )
        print(col)
        print("-" * len(col))
        for i, c in enumerate(candidates, 1):
            rr_display = f"{c['reward_risk']:.2f}" if math.isfinite(c["reward_risk"]) else "N/A"
            ok = "YES" if c["eligible"] else "no"
            print(
                f"{i:>3}  {rr_display:>7}  {c['net_debit']:>8.4f}  {c['max_profit']:>9.4f}  {c['wing_width']:>6.2f}  "
                f"{c['body_richness']:>7.2f}  {c['wing_cheapness']:>8.2f}  {c['body_distance_pct']*100:>6.2f}  "
                f"{c['lower_symbol']:<16}  {c['middle_symbol']:<16}  {c['upper_symbol']:<16}  {ok}"
            )

        eligible = [c for c in candidates if c["eligible"]]
        best = eligible[0] if eligible else candidates[0]

        cross_asset_results.append({
            "asset":     current_asset,
            "garch_vol": garch_vol,
            "spot":      spot,
            "best":      best,
            "eligible":  bool(eligible),
        })

        print(f"\n{'='*88}")
        label = "BEST ELIGIBLE" if eligible else "BEST AVAILABLE (none met all thresholds)"
        print(f"[{current_asset}] {label} LONG CALL BUTTERFLY:")
        print(
            f"  BUY  {best['lower_symbol']:<20} strike={best['lower_strike']:.2f}  Δ={best['lower_delta']:.2f}  "
            f"IV={best['lower_iv']:.2f}%  ask={best['lower_ask']:.4f}"
        )
        print(
            f"  SELL 2x {best['middle_symbol']:<17} strike={best['middle_strike']:.2f}  Δ={best['middle_delta']:.2f}  "
            f"IV={best['middle_iv']:.2f}%  bid={best['middle_bid']:.4f}"
        )
        print(
            f"  BUY  {best['upper_symbol']:<20} strike={best['upper_strike']:.2f}  Δ={best['upper_delta']:.2f}  "
            f"IV={best['upper_iv']:.2f}%  ask={best['upper_ask']:.4f}"
        )
        rr_summary = f"{best['reward_risk']:.2f}" if math.isfinite(best["reward_risk"]) else "N/A"
        print(
            f"  Net debit          : {best['net_debit']:.4f}  |  Max profit : {best['max_profit']:.4f}  "
            f"|  Max loss : {best['max_loss']:.4f}  |  Reward/Risk : {rr_summary}"
        )
        print(
            f"  Body richness      : {best['body_richness']:.2f}pp vs GARCH  |  "
            f"Wing cheapness : {best['wing_cheapness']:.2f}pp  |  "
            f"Body distance : {best['body_distance_pct']*100:.2f}% from spot"
        )
        print(f"{'='*88}\n")

    if len(cross_asset_results) < 2:
        return

    cross_asset_results.sort(key=lambda r: (r["eligible"], r["best"]["score"]), reverse=True)

    print(f"\n{'='*120}")
    print("CROSS-ASSET LONG CALL BUTTERFLY RANKING")
    print(f"{'='*120}")
    col2 = (
        f"{'#':>3}  {'Asset':<8}  {'RR':>7}  {'Debit':>8}  {'MaxProfit':>9}  "
        f"{'BdRich':>7}  {'WgCheap':>8}  {'Dist%':>6}  {'GARCHvol':>10}  {'BODY':<20}  OK"
    )
    print(col2)
    print("-" * len(col2))
    for i, r in enumerate(cross_asset_results, 1):
        b = r["best"]
        rr_display = f"{b['reward_risk']:.2f}" if math.isfinite(b["reward_risk"]) else "N/A"
        ok = "YES" if r["eligible"] else "no"
        print(
            f"{i:>3}  {r['asset']:<8}  {rr_display:>7}  {b['net_debit']:>8.4f}  {b['max_profit']:>9.4f}  "
            f"{b['body_richness']:>7.2f}  {b['wing_cheapness']:>8.2f}  {b['body_distance_pct']*100:>6.2f}  "
            f"{r['garch_vol']:>10.2f}  {b['middle_symbol']:<20}  {ok}"
        )

    overall = cross_asset_results[0]
    ob = overall["best"]
    print(f"\n{'='*120}")
    status = "ELIGIBLE" if overall["eligible"] else "AVAILABLE (below thresholds)"
    print(
        f"OVERALL BEST {status}:  {overall['asset']}  |  "
        f"GARCH={overall['garch_vol']:.2f}%  |  spot={overall['spot']:.2f}"
    )
    print(
        f"  BUY  {ob['lower_symbol']:<20} strike={ob['lower_strike']:.2f}  Δ={ob['lower_delta']:.2f}  "
        f"IV={ob['lower_iv']:.2f}%  ask={ob['lower_ask']:.4f}"
    )
    print(
        f"  SELL 2x {ob['middle_symbol']:<17} strike={ob['middle_strike']:.2f}  Δ={ob['middle_delta']:.2f}  "
        f"IV={ob['middle_iv']:.2f}%  bid={ob['middle_bid']:.4f}"
    )
    print(
        f"  BUY  {ob['upper_symbol']:<20} strike={ob['upper_strike']:.2f}  Δ={ob['upper_delta']:.2f}  "
        f"IV={ob['upper_iv']:.2f}%  ask={ob['upper_ask']:.4f}"
    )
    rr_summary = f"{ob['reward_risk']:.2f}" if math.isfinite(ob["reward_risk"]) else "N/A"
    print(
        f"  Net debit          : {ob['net_debit']:.4f}  |  Max profit : {ob['max_profit']:.4f}  "
        f"|  Max loss : {ob['max_loss']:.4f}  |  Reward/Risk : {rr_summary}"
    )
    print(
        f"  Body richness      : {ob['body_richness']:.2f}pp vs GARCH  |  "
        f"Wing cheapness : {ob['wing_cheapness']:.2f}pp  |  "
        f"Body distance : {ob['body_distance_pct']*100:.2f}% from spot"
    )
    print(f"{'='*120}\n")


# ══════════════════════════════════════════════════════════════════════════════
# IRON CONDOR SCANNER
# Structure: BUY lp  +  SELL sp  +  SELL sc  +  BUY lc  (net credit)
#   lp_strike < sp_strike < spot < sc_strike < lc_strike
# Profit if spot stays between the two short strikes at expiry.
#
# Probability of profit (PoP) is approximated via the short-leg deltas:
#   PoP ≈ 1 − sc_delta − |sp_delta|
# This is the standard "delta ≈ P(ITM)" shortcut used in practice.
#
# Scoring: PoP×100  +  RR×10  +  call_iv_edge  +  put_iv_edge  −  |ΔNet|×5
# ══════════════════════════════════════════════════════════════════════════════

def build_iron_condor_candidates(utils, calls_dict, puts_dict, spot_price, garch_vol):
    """
    Build and rank all iron condor candidates from the live option chain.
    Returns a list sorted by (eligible, score) descending.
    """
    # ── 1. Filter by delta role ───────────────────────────────────────────────
    short_calls = {d: v for d, v in calls_dict.items()
                   if IRON_CONDOR_SHORT_DELTA_MIN <= d <= IRON_CONDOR_SHORT_DELTA_MAX}
    wing_calls  = {d: v for d, v in calls_dict.items()
                   if d < IRON_CONDOR_SHORT_DELTA_MIN}   # further OTM calls (protection)

    short_puts  = {d: v for d, v in puts_dict.items()
                   if IRON_CONDOR_SHORT_DELTA_MIN <= abs(d) <= IRON_CONDOR_SHORT_DELTA_MAX}
    wing_puts   = {d: v for d, v in puts_dict.items()
                   if abs(d) < IRON_CONDOR_SHORT_DELTA_MIN}  # further OTM puts (protection)

    if not short_calls or not wing_calls or not short_puts or not wing_puts:
        return []

    # ── 2. Pre-fetch live quotes (one call per symbol, cached) ────────────────
    all_names = set()
    for data in list(short_calls.values()) + list(wing_calls.values()) + \
                list(short_puts.values()) + list(wing_puts.values()):
        all_names.add(data["option_name"])

    quote_cache = {}
    for name in all_names:
        info = utils.get_option_info_with_quote(name)
        if info is not None and info.bid > 0 and info.ask > 0:
            quote_cache[name] = info

    # ── 3. Iterate all 4-leg combinations ─────────────────────────────────────
    candidates = []

    for sc_delta, sc_data in short_calls.items():
        sc_name = sc_data["option_name"]
        if sc_name not in quote_cache:
            continue
        sc_quote = quote_cache[sc_name]

        for lc_delta, lc_data in wing_calls.items():
            # Long call must be further OTM (higher strike) than short call
            if lc_data["strike"] <= sc_data["strike"]:
                continue
            lc_name = lc_data["option_name"]
            if lc_name not in quote_cache:
                continue
            lc_quote = quote_cache[lc_name]

            call_wing_width   = lc_data["strike"] - sc_data["strike"]
            if call_wing_width < IRON_CONDOR_MIN_WING_WIDTH:
                continue
            call_spread_credit = sc_quote.bid - lc_quote.ask
            if call_spread_credit <= 0:
                continue  # debit call spread — skip

            for sp_delta, sp_data in short_puts.items():
                # Short put must be below spot and below short call
                if sp_data["strike"] >= sc_data["strike"]:
                    continue
                sp_name = sp_data["option_name"]
                if sp_name not in quote_cache:
                    continue
                sp_quote = quote_cache[sp_name]

                for lp_delta, lp_data in wing_puts.items():
                    # Long put must be further OTM (lower strike) than short put
                    if lp_data["strike"] >= sp_data["strike"]:
                        continue
                    lp_name = lp_data["option_name"]
                    if lp_name not in quote_cache:
                        continue
                    lp_quote = quote_cache[lp_name]

                    put_wing_width   = sp_data["strike"] - lp_data["strike"]
                    if put_wing_width < IRON_CONDOR_MIN_WING_WIDTH:
                        continue
                    put_spread_credit = sp_quote.bid - lp_quote.ask
                    if put_spread_credit <= 0:
                        continue  # debit put spread — skip

                    # ── Metrics ───────────────────────────────────────────────
                    net_credit      = call_spread_credit + put_spread_credit
                    max_loss_call   = call_wing_width - call_spread_credit
                    max_loss_put    = put_wing_width  - put_spread_credit
                    max_loss        = max(max_loss_call, max_loss_put)
                    rr              = net_credit / max_loss if max_loss > 0 else -math.inf

                    # Delta-based PoP: probability spot stays between short strikes
                    pop             = 1.0 - sc_delta - abs(sp_delta)

                    call_iv_edge    = sc_data["iv"] - garch_vol
                    put_iv_edge     = sp_data["iv"] - garch_vol
                    delta_imbalance = abs(sc_delta + sp_delta)  # sp_delta is negative
                    call_dist_pct   = (sc_data["strike"] - spot_price) / max(spot_price, 1.0)
                    put_dist_pct    = (spot_price - sp_data["strike"]) / max(spot_price, 1.0)

                    eligible = (
                        pop             >= IRON_CONDOR_MIN_POP
                        and rr          >= IRON_CONDOR_MIN_REWARD_RISK
                        and call_iv_edge >= IRON_CONDOR_MIN_IV_EDGE
                        and put_iv_edge  >= IRON_CONDOR_MIN_IV_EDGE
                        and delta_imbalance <= IRON_CONDOR_MAX_DELTA_IMBALANCE
                    )

                    # Score: PoP is the primary driver, then R/R, then IV edge
                    score = (
                        pop * 100
                        + rr * 10
                        + call_iv_edge
                        + put_iv_edge
                        - delta_imbalance * 5
                    )

                    candidates.append({
                        "score":              score,
                        "eligible":           eligible,
                        "net_credit":         net_credit,
                        "max_loss":           max_loss,
                        "rr":                 rr,
                        "pop":                pop,
                        "call_spread_credit": call_spread_credit,
                        "put_spread_credit":  put_spread_credit,
                        "call_wing_width":    call_wing_width,
                        "put_wing_width":     put_wing_width,
                        "max_loss_call":      max_loss_call,
                        "max_loss_put":       max_loss_put,
                        "call_iv_edge":       call_iv_edge,
                        "put_iv_edge":        put_iv_edge,
                        "delta_imbalance":    delta_imbalance,
                        "call_dist_pct":      call_dist_pct,
                        "put_dist_pct":       put_dist_pct,
                        "sc_symbol":          sc_name,
                        "sc_strike":          sc_data["strike"],
                        "sc_delta":           sc_delta,
                        "sc_iv":              sc_data["iv"],
                        "sc_bid":             sc_quote.bid,
                        "lc_symbol":          lc_name,
                        "lc_strike":          lc_data["strike"],
                        "lc_delta":           lc_delta,
                        "lc_iv":              lc_data["iv"],
                        "lc_ask":             lc_quote.ask,
                        "sp_symbol":          sp_name,
                        "sp_strike":          sp_data["strike"],
                        "sp_delta":           sp_delta,
                        "sp_iv":              sp_data["iv"],
                        "sp_bid":             sp_quote.bid,
                        "lp_symbol":          lp_name,
                        "lp_strike":          lp_data["strike"],
                        "lp_delta":           lp_delta,
                        "lp_iv":              lp_data["iv"],
                        "lp_ask":             lp_quote.ask,
                        "garch_vol":          garch_vol,
                    })

    return sorted(candidates, key=lambda c: (c["eligible"], c["score"]), reverse=True)


async def scan_iron_condor_opportunities(
    asset: str = None,
    expiry_rank: int = IRON_CONDOR_EXPIRY_RANK,
):
    """
    Scan and rank iron condor candidates for all assets in ASSET_SYMBOL.
    When asset is not None, scans only that asset.
    No orders are placed.
    """
    mt5_conn   = MT5Connector()
    quant_calc = QuantCalculation()
    utils      = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    assets_to_scan      = [asset] if asset is not None else ASSET_SYMBOL
    cross_asset_results = []

    for current_asset in assets_to_scan:
        if not mt5_conn.symbol_select(current_asset, True):
            print(f"[{current_asset}] Failed to select — skipping")
            continue

        tick_ok = False
        for _ in range(10):
            tick = mt5_conn.get_mt5_connector().symbol_info_tick(current_asset)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                tick_ok = True
                break
            print(f"Waiting for {current_asset} tick data...")
            await asyncio.sleep(1)
        if not tick_ok:
            print(f"{current_asset} has no tick data after 10 s — skipping")
            continue

        symbol_info = mt5_conn.get_symbol_info(current_asset)
        spot = (symbol_info.bid + symbol_info.ask) / 2

        print(f"\n{'='*96}")
        print(f"IRON CONDOR SCANNER  |  {current_asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
        print(f"{'='*96}")

        spot_prices_data = mt5_conn.get_data(
            current_asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        print(f"GARCH vol : {garch_vol:.2f}%")

        chain_options = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=expiry_rank
        )
        if not chain_options:
            print("No option chain returned for selected expiry rank")
            continue

        expiration_time = next(iter(chain_options.keys()))
        print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

        calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not calls_dict or not puts_dict:
            print("No calls or puts returned from chain")
            continue

        candidates = build_iron_condor_candidates(utils, calls_dict, puts_dict, spot, garch_vol)
        if not candidates:
            print("No iron condor candidates found with valid live quotes.")
            continue

        print(f"\n{'='*148}")
        print(
            "RANKED IRON CONDOR CANDIDATES  "
            f"(min PoP >= {IRON_CONDOR_MIN_POP*100:.0f}%  |  "
            f"min R/R >= {IRON_CONDOR_MIN_REWARD_RISK:.2f}  |  "
            f"min IV edge >= {IRON_CONDOR_MIN_IV_EDGE:.1f}pp  |  "
            f"max |ΔNet| <= {IRON_CONDOR_MAX_DELTA_IMBALANCE:.2f}  |  "
            f"GARCH={garch_vol:.2f}%)"
        )
        print(f"{'='*148}")
        col = (
            f"{'#':>3}  {'PoP%':>5}  {'R/R':>5}  {'Credit':>7}  {'MaxLoss':>8}  "
            f"{'CEdge':>6}  {'PEdge':>6}  {'|ΔNet|':>6}  "
            f"{'SELL_C':<14}  {'BUY_C':<14}  {'SELL_P':<14}  {'BUY_P':<14}  OK"
        )
        print(col)
        print("-" * len(col))
        for i, c in enumerate(candidates[:25], 1):
            rr_display = f"{c['rr']:.2f}" if math.isfinite(c["rr"]) else "N/A"
            ok = "YES" if c["eligible"] else "no"
            print(
                f"{i:>3}  {c['pop']*100:>5.1f}  {rr_display:>5}  {c['net_credit']:>7.4f}  {c['max_loss']:>8.4f}  "
                f"{c['call_iv_edge']:>6.2f}  {c['put_iv_edge']:>6.2f}  {c['delta_imbalance']:>6.4f}  "
                f"{c['sc_symbol']:<14}  {c['lc_symbol']:<14}  {c['sp_symbol']:<14}  {c['lp_symbol']:<14}  {ok}"
            )

        eligible = [c for c in candidates if c["eligible"]]
        best     = eligible[0] if eligible else candidates[0]

        cross_asset_results.append({
            "asset":    current_asset,
            "garch_vol": garch_vol,
            "spot":     spot,
            "best":     best,
            "eligible": bool(eligible),
        })

        print(f"\n{'='*96}")
        label = "BEST ELIGIBLE" if eligible else "BEST AVAILABLE (none met all thresholds)"
        print(f"[{current_asset}] {label} IRON CONDOR:")
        print(f"  BUY  {best['lp_symbol']:<18} strike={best['lp_strike']:.2f}  Δ={best['lp_delta']:.2f}  IV={best['lp_iv']:.2f}%  ask={best['lp_ask']:.4f}")
        print(f"  SELL {best['sp_symbol']:<18} strike={best['sp_strike']:.2f}  Δ={best['sp_delta']:.2f}  IV={best['sp_iv']:.2f}%  bid={best['sp_bid']:.4f}  edge={best['put_iv_edge']:+.2f}pp")
        print(f"  SELL {best['sc_symbol']:<18} strike={best['sc_strike']:.2f}  Δ={best['sc_delta']:.2f}  IV={best['sc_iv']:.2f}%  bid={best['sc_bid']:.4f}  edge={best['call_iv_edge']:+.2f}pp")
        print(f"  BUY  {best['lc_symbol']:<18} strike={best['lc_strike']:.2f}  Δ={best['lc_delta']:.2f}  IV={best['lc_iv']:.2f}%  ask={best['lc_ask']:.4f}")
        rr_summary = f"{best['rr']:.2f}" if math.isfinite(best["rr"]) else "N/A"
        print(f"  Net credit   : {best['net_credit']:.4f}  |  Max loss : {best['max_loss']:.4f}  |  R/R : {rr_summary}")
        print(f"  PoP          : {best['pop']*100:.1f}%  (delta approx)")
        print(f"  Call spread  : {best['sc_strike']:.2f}–{best['lc_strike']:.2f}  width={best['call_wing_width']:.2f}  credit={best['call_spread_credit']:.4f}  ({best['call_dist_pct']*100:.2f}% above spot)")
        print(f"  Put spread   : {best['lp_strike']:.2f}–{best['sp_strike']:.2f}  width={best['put_wing_width']:.2f}  credit={best['put_spread_credit']:.4f}  ({best['put_dist_pct']*100:.2f}% below spot)")
        print(f"  |ΔNet|       : {best['delta_imbalance']:.4f}  |  GARCH : {best['garch_vol']:.2f}%")
        print(f"{'='*96}\n")

    if len(cross_asset_results) < 2:
        return

    cross_asset_results.sort(key=lambda r: (r["eligible"], r["best"]["score"]), reverse=True)

    print(f"\n{'='*110}")
    print("CROSS-ASSET IRON CONDOR RANKING")
    print(f"{'='*110}")
    col2 = (
        f"{'#':>3}  {'Asset':<8}  {'PoP%':>5}  {'R/R':>5}  {'Credit':>7}  {'MaxLoss':>8}  "
        f"{'CEdge':>6}  {'PEdge':>6}  {'GARCHvol':>9}  {'SELL_C':<16}  {'SELL_P':<16}  OK"
    )
    print(col2)
    print("-" * len(col2))
    for i, r in enumerate(cross_asset_results, 1):
        b  = r["best"]
        rr = f"{b['rr']:.2f}" if math.isfinite(b["rr"]) else "N/A"
        ok = "YES" if r["eligible"] else "no"
        print(
            f"{i:>3}  {r['asset']:<8}  {b['pop']*100:>5.1f}  {rr:>5}  {b['net_credit']:>7.4f}  {b['max_loss']:>8.4f}  "
            f"{b['call_iv_edge']:>6.2f}  {b['put_iv_edge']:>6.2f}  {r['garch_vol']:>9.2f}  "
            f"{b['sc_symbol']:<16}  {b['sp_symbol']:<16}  {ok}"
        )

    overall = cross_asset_results[0]
    ob      = overall["best"]
    rr_s    = f"{ob['rr']:.2f}" if math.isfinite(ob["rr"]) else "N/A"
    status  = "ELIGIBLE" if overall["eligible"] else "AVAILABLE (below thresholds)"
    print(f"\n{'='*110}")
    print(f"OVERALL BEST {status}:  {overall['asset']}  |  GARCH={overall['garch_vol']:.2f}%  |  spot={overall['spot']:.2f}")
    print(f"  BUY  {ob['lp_symbol']:<18} strike={ob['lp_strike']:.2f}  Δ={ob['lp_delta']:.2f}  IV={ob['lp_iv']:.2f}%  ask={ob['lp_ask']:.4f}")
    print(f"  SELL {ob['sp_symbol']:<18} strike={ob['sp_strike']:.2f}  Δ={ob['sp_delta']:.2f}  IV={ob['sp_iv']:.2f}%  bid={ob['sp_bid']:.4f}  edge={ob['put_iv_edge']:+.2f}pp")
    print(f"  SELL {ob['sc_symbol']:<18} strike={ob['sc_strike']:.2f}  Δ={ob['sc_delta']:.2f}  IV={ob['sc_iv']:.2f}%  bid={ob['sc_bid']:.4f}  edge={ob['call_iv_edge']:+.2f}pp")
    print(f"  BUY  {ob['lc_symbol']:<18} strike={ob['lc_strike']:.2f}  Δ={ob['lc_delta']:.2f}  IV={ob['lc_iv']:.2f}%  ask={ob['lc_ask']:.4f}")
    print(f"  Net credit : {ob['net_credit']:.4f}  |  Max loss : {ob['max_loss']:.4f}  |  R/R : {rr_s}  |  PoP : {ob['pop']*100:.1f}%")
    print(f"{'='*110}\n")


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY SKEW SCANNER
#
# Measures the put-call IV spread and skew slope across delta points, then
# identifies three structured trades per asset:
#
#   1. Risk Reversal   : SELL ~25Δ put  + BUY ~25Δ call
#      Edge: put_iv − call_iv (put richness)
#
#   2. Skew Put Spread : BUY ~35Δ put + SELL ~15Δ put
#      Edge: far-OTM put overpriced by skew → cheaper net debit
#
#   3. Skew Call Spread: SELL ~35Δ call + BUY ~15Δ call  (bear call spread)
#      Edge: near-OTM call overpriced → collect credit
#
# Scoring: rr_pp * 2  +  put_skew_slope  +  rr_net_credit * 10
# ══════════════════════════════════════════════════════════════════════════════

def _skew_nearest(delta_dict, target_delta, tol):
    """Return (delta_key, data) for the entry closest to target_delta within tol."""
    if not delta_dict:
        return None, None
    best = min(delta_dict.keys(), key=lambda d: abs(d - target_delta))
    return (best, delta_dict[best]) if abs(best - target_delta) <= tol else (None, None)


def compute_skew_metrics(utils, calls_dict, puts_dict, garch_vol, spot, T):
    """
    Two-trade volatility skew strategy:

      Trade 1 — Cheapest straddle (BUY call + BUY put at the same strike):
        Filter calls with delta in [0.25, 0.75].  For each such call, find the
        put at the exact same strike.  Pick the strike with the lowest average
        (call_iv + put_iv) / 2.  BUY both legs.
        Eligible when straddle_avg_iv < garch_vol − SKEW_MIN_RR_PP
        (implied vol cheaper than realised vol).

      Trade 2 — 2σ_Δ call spread, minimum |sell_IV − buy_IV| (SELL near / BUY far):
        Compute σ_Δ = std_dev of all available call deltas.
        Target delta separation = 2 × σ_Δ.
        Among all (short_call, long_call) pairs, keep those whose
        delta separation is within ±50 % of the target (relax to closest 10).
        From that set pick the pair with minimum |sell_IV − buy_IV|.
        Eligible when net_credit > 0.

    Returns a metrics dict or None if neither leg has a live quote.
    """
    # ── 1. IV surface for context display ────────────────────────────────────
    surface = {}
    for target in sorted({0.10, SKEW_FAR_DELTA, SKEW_RR_DELTA, SKEW_NEAR_DELTA, 0.45}):
        cd, cd_data = _skew_nearest(calls_dict, target, SKEW_DELTA_TOLERANCE)
        pd, pd_data = _skew_nearest(puts_dict, -target, SKEW_DELTA_TOLERANCE)
        if cd_data and pd_data:
            surface[target] = {
                "call_delta":  cd,  "call_iv": cd_data["iv"],
                "call_option": cd_data["option_name"], "call_strike": cd_data["strike"],
                "put_delta":   pd,  "put_iv":  pd_data["iv"],
                "put_option":  pd_data["option_name"], "put_strike":  pd_data["strike"],
                "rr": cd_data["iv"] - pd_data["iv"],
            }

    # ── 2. Trade 1: BUY cheapest straddle within delta [0.25, 0.75] ──────────
    # Collect calls in delta range; for each find the put at the same strike.
    straddle_call_info = straddle_put_info = None
    straddle_eligible  = False
    straddle_avg_iv    = None
    straddle_net_delta = None     # call_Δ + put_Δ  (≈ 0 when ATM)

    valid_calls = [(d, v) for d, v in calls_dict.items() if 0.25 <= d <= 0.75]

    # Build put strike index for fast lookup
    put_by_strike = {}
    for pd_key, pd_data in puts_dict.items():
        put_by_strike[pd_data["strike"]] = (pd_key, pd_data)

    # Find the strike with the lowest average straddle IV
    best_avg_iv   = float("inf")
    best_c_delta  = best_c_data = best_p_delta = best_p_data = None
    for c_delta, c_data in valid_calls:
        c_strike = c_data["strike"]
        # Exact strike match; fall back to closest within 0.5 % of spot
        if c_strike in put_by_strike:
            p_delta, p_data = put_by_strike[c_strike]
        else:
            closest_k = min(put_by_strike.keys(), key=lambda k: abs(k - c_strike))
            if abs(closest_k - c_strike) > 0.005 * spot:
                continue
            p_delta, p_data = put_by_strike[closest_k]
        avg_iv = (c_data["iv"] + p_data["iv"]) / 2.0
        if avg_iv < best_avg_iv:
            best_avg_iv  = avg_iv
            best_c_delta = c_delta
            best_c_data  = c_data
            best_p_delta = p_delta
            best_p_data  = p_data

    if best_c_data and best_p_data:
        cq = utils.get_option_info_with_quote(best_c_data["option_name"])
        pq = utils.get_option_info_with_quote(best_p_data["option_name"])
        if (cq and cq.bid > 0 and cq.ask > 0
                and pq and pq.bid > 0 and pq.ask > 0):
            straddle_avg_iv    = best_avg_iv
            straddle_net_delta = best_c_delta + best_p_delta   # net portfolio Δ
            straddle_eligible  = (garch_vol - straddle_avg_iv) >= SKEW_MIN_RR_PP
            straddle_call_info = {
                "option": best_c_data["option_name"], "strike": best_c_data["strike"],
                "delta":  best_c_delta, "iv":  best_c_data["iv"],
                "bid":    cq.bid,       "ask": cq.ask,
            }
            straddle_put_info = {
                "option": best_p_data["option_name"], "strike": best_p_data["strike"],
                "delta":  best_p_delta, "iv":  best_p_data["iv"],
                "bid":    pq.bid,       "ask": pq.ask,
            }

    # ── 3. Trade 2: Call spread ~2σ_Δ wide, minimum |sell_IV − buy_IV| ───────
    call_delta_list = list(calls_dict.keys())
    sigma_delta     = float(np.std(call_delta_list)) if len(call_delta_list) > 1 else 0.10
    target_sep      = 2.0 * sigma_delta              # target delta separation

    call_deltas = sorted(calls_dict.keys(), reverse=True)
    all_pairs   = []
    for i, sd in enumerate(call_deltas):
        for ld in call_deltas[i + 1:]:              # ld < sd guaranteed
            delta_sep = sd - ld
            iv_diff   = abs(calls_dict[sd]["iv"] - calls_dict[ld]["iv"])
            all_pairs.append((sd, ld, delta_sep, iv_diff))

    # Filter to pairs near 2σ_Δ; relax to closest 10 if nothing qualifies.
    filtered = [p for p in all_pairs if abs(p[2] - target_sep) <= 0.5 * target_sep]
    if not filtered:
        filtered = sorted(all_pairs, key=lambda p: abs(p[2] - target_sep))[:10]

    filtered.sort(key=lambda p: p[3])   # minimum |IV diff| first

    cs_short_info = cs_long_info = None
    cs_net_credit = cs_iv_diff = cs_delta_sep = cs_ratio = None
    cs_eligible   = False

    for sd, ld, delta_sep, iv_diff in filtered:
        sd_data = calls_dict[sd]
        ld_data = calls_dict[ld]
        sq = utils.get_option_info_with_quote(sd_data["option_name"])
        lq = utils.get_option_info_with_quote(ld_data["option_name"])
        if not (sq and sq.bid > 0 and sq.ask > 0 and lq and lq.bid > 0 and lq.ask > 0):
            continue
        # ── Delta-neutral ratio: SELL 1, BUY (Δ_sell / Δ_buy) of the long leg ──
        # net_delta = ratio × Δ_buy − 1 × Δ_sell  = 0  by construction
        _ratio        = sd / ld if ld > 0 else 1.0
        cs_ratio      = round(_ratio, 4)
        cs_net_credit = sq.bid - cs_ratio * lq.ask
        cs_iv_diff    = iv_diff
        cs_delta_sep  = delta_sep
        cs_eligible   = cs_net_credit >= 0
        cs_short_info = {
            "option": sd_data["option_name"], "strike": sd_data["strike"],
            "delta":  sd,                     "iv":     sd_data["iv"],
            "bid":    sq.bid,                 "ask":    sq.ask,
        }
        cs_long_info = {
            "option": ld_data["option_name"], "strike": ld_data["strike"],
            "delta":  ld,                     "iv":     ld_data["iv"],
            "bid":    lq.bid,                 "ask":    lq.ask,
        }
        break

    if not straddle_call_info and not cs_short_info:
        return None

    # ── 4. Composite score (higher = more attractive) ────────────────────────
    # Straddle cheapness: how far IV is below GARCH.
    skew_score = (garch_vol - straddle_avg_iv) if straddle_avg_iv is not None else 0.0

    return {
        "surface":            surface,
        "straddle_call":       straddle_call_info,
        "straddle_put":        straddle_put_info,
        "straddle_avg_iv":     straddle_avg_iv,
        "straddle_net_delta":  straddle_net_delta,   # call_Δ + put_Δ
        "straddle_eligible":   straddle_eligible,
        "cs_short":            cs_short_info,
        "cs_long":             cs_long_info,
        "cs_ratio":            cs_ratio,             # SELL 1 : BUY cs_ratio
        "cs_net_credit":       cs_net_credit,
        "cs_iv_diff":          cs_iv_diff,
        "cs_delta_sep":        cs_delta_sep,
        "cs_target_delta_sep": target_sep,
        "sigma_delta":         sigma_delta,
        "cs_eligible":         cs_eligible,
        "skew_score":          skew_score,
        "garch_vol":           garch_vol,
    }


async def scan_volatility_skew_opportunities(
    asset: str = None,
    expiry_rank: int = SKEW_EXPIRY_RANK,
):
    """
    Scan the volatility skew surface for all assets in ASSET_SYMBOL.
    When asset is not None, scans only that asset.  No orders are placed.

    Per asset outputs:
      • Full IV surface table (call/put IV at fixed delta points)
      • TRADE 1 — Cheapest straddle : BUY call + put at same strike, lowest avg IV in Δ [0.25, 0.75]
      • TRADE 2 — 2σ_Δ call spread  : SELL/BUY pair with delta sep ≈ 2×std_dev(Δ), min |IV diff|
    Ends with a cross-asset ranking table.
    """
    mt5_conn   = MT5Connector()
    quant_calc = QuantCalculation()
    utils      = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    assets_to_scan      = [asset] if asset is not None else ASSET_SYMBOL
    cross_asset_results = []

    for current_asset in assets_to_scan:
        if not mt5_conn.symbol_select(current_asset, True):
            print(f"[{current_asset}] Failed to select — skipping")
            continue

        tick_ok = False
        for _ in range(10):
            tick = mt5_conn.get_mt5_connector().symbol_info_tick(current_asset)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                tick_ok = True
                break
            print(f"Waiting for {current_asset} tick data...")
            await asyncio.sleep(1)
        if not tick_ok:
            print(f"{current_asset} has no tick data after 10 s — skipping")
            continue

        symbol_info = mt5_conn.get_symbol_info(current_asset)
        spot = (symbol_info.bid + symbol_info.ask) / 2

        print(f"\n{'='*96}")
        print(f"VOLATILITY SKEW SCANNER  |  {current_asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
        print(f"{'='*96}")

        spot_prices_data = mt5_conn.get_data(
            current_asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        print(f"GARCH vol : {garch_vol:.2f}%")

        chain_options = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=expiry_rank
        )
        if not chain_options:
            print("No option chain returned for selected expiry rank")
            continue

        expiration_time = next(iter(chain_options.keys()))
        print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

        calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not calls_dict or not puts_dict:
            print("No calls or puts returned from chain")
            continue

        _, T = utils.get_factor_from_expiration_time(expiration_time)

        metrics = compute_skew_metrics(utils, calls_dict, puts_dict, garch_vol, spot, T)
        if metrics is not None:
            print(f"Δ std_dev : {metrics['sigma_delta']:.4f}  |  2σ_Δ target sep : {metrics['cs_target_delta_sep']:.4f}\n")
        if metrics is None:
            print("No tradeable legs found (no live quotes)")
            continue

        # ── IV surface table ──────────────────────────────────────────────────
        print(f"{'Δ':>6}  {'CallIV%':>8}  {'PutIV%':>7}  {'RR (C−P)':>9}  {'CallOpt':<20}  {'PutOpt':<20}")
        print("-" * 90)
        for tgt in sorted(metrics["surface"]):
            row = metrics["surface"][tgt]
            skew_tag = "  ← steep" if row["rr"] < -SKEW_MIN_RR_PP else ""
            print(
                f"{tgt:>6.2f}  {row['call_iv']:>8.2f}  {row['put_iv']:>7.2f}  "
                f"{row['rr']:>+9.2f}  {row['call_option']:<20}  {row['put_option']:<20}{skew_tag}"
            )

        # ── Trade 1: Cheapest straddle (BUY call + BUY put, same strike) ────────
        print(f"\n{'─'*80}")
        print("  TRADE 1 — CHEAPEST STRADDLE   BUY call + BUY put at same strike")
        print("             (lowest avg IV within call Δ [0.25, 0.75])")
        if metrics["straddle_call"] and metrics["straddle_put"]:
            sc  = metrics["straddle_call"]
            sp  = metrics["straddle_put"]
            tag = "  ← ELIGIBLE" if metrics["straddle_eligible"] else ""
            avg = metrics["straddle_avg_iv"]
            ivg = garch_vol - avg
            nd  = metrics["straddle_net_delta"]
            print(f"    BUY  {sc['option']:<20} K={sc['strike']:.2f}  Δ={sc['delta']:+.4f}  "
                  f"IV={sc['iv']:.2f}%  ask={sc['ask']:.4f}")
            print(f"    BUY  {sp['option']:<20} K={sp['strike']:.2f}  Δ={sp['delta']:+.4f}  "
                  f"IV={sp['iv']:.2f}%  ask={sp['ask']:.4f}")
            print(f"    Avg straddle IV : {avg:.2f}%  |  GARCH−IV : {ivg:+.2f}pp{tag}")
            print(f"    Net portfolio Δ : {nd:+.4f}  "
                  + (f"→ delta-neutral" if abs(nd) < 0.05
                     else f"→ hedge: {'SELL' if nd > 0 else 'BUY'} {abs(nd):.4f} units of underlying"))
        else:
            print("    No valid quotes found for a straddle in delta [0.25, 0.75]")

        # ── Trade 2: 2σ_Δ call spread, min IV diff ────────────────────────────
        print(f"\n{'─'*80}")
        print(f"  TRADE 2 — 2σ_Δ CALL SPREAD   SELL/BUY pair  "
              f"target ΔΔ={metrics['cs_target_delta_sep']:.4f}  (2 × σ_Δ={metrics['sigma_delta']:.4f})")
        if metrics["cs_short"] and metrics["cs_long"]:
            csh   = metrics["cs_short"]
            clg   = metrics["cs_long"]
            ratio = metrics["cs_ratio"]
            tag   = "  ← ELIGIBLE" if metrics["cs_eligible"] else ""
            nc    = f"{metrics['cs_net_credit']:.4f}" if metrics["cs_net_credit"] is not None else "N/A"
            ivd   = f"{metrics['cs_iv_diff']:.2f}pp"  if metrics["cs_iv_diff"]    is not None else "N/A"
            ads   = f"{metrics['cs_delta_sep']:.4f}"  if metrics["cs_delta_sep"]   is not None else "N/A"
            print(f"    Ratio : SELL 1 × {csh['option']:<20}  BUY {ratio:.4f} × {clg['option']:<20}")
            print(f"    (ratio = Δ_sell/Δ_buy = {csh['delta']:.4f}/{clg['delta']:.4f} — net portfolio Δ = 0)")
            print(f"    Actual ΔΔ : {ads}  (target : {metrics['cs_target_delta_sep']:.4f})")
            print(f"    SELL {csh['option']:<20} K={csh['strike']:.2f}  Δ={csh['delta']:+.4f}  "
                  f"IV={csh['iv']:.2f}%  bid={csh['bid']:.4f}")
            print(f"    BUY  {clg['option']:<20} K={clg['strike']:.2f}  Δ={clg['delta']:+.4f}  "
                  f"IV={clg['iv']:.2f}%  ask={clg['ask']:.4f}")
            print(f"    Net credit (ratio-adjusted) : {nc}  |  |sell_IV − buy_IV| : {ivd}{tag}")
        else:
            print("    No valid quotes found for the 2σ_Δ call spread")

        print(f"\n{'='*96}\n")

        cross_asset_results.append({
            "asset":     current_asset,
            "garch_vol": garch_vol,
            "spot":      spot,
            "metrics":   metrics,
            "any_eligible": metrics["straddle_eligible"] or metrics["cs_eligible"],
        })

    if len(cross_asset_results) < 2:
        return

    cross_asset_results.sort(
        key=lambda r: (r["any_eligible"], r["metrics"]["skew_score"]),
        reverse=True,
    )

    print(f"\n{'='*120}")
    print("CROSS-ASSET VOLATILITY SKEW RANKING")
    print(f"{'='*120}")
    hdr = (
        f"{'#':>3}  {'Asset':<8}  {'StraddleIV%':>12}  {'GARCH−IV':>9}  "
        f"{'σ_Δ':>6}  {'ActΔΔ':>7}  {'IVdiff':>7}  {'CSCredit':>9}  {'GARCH%':>7}  "
        f"{'STR?':>5}  {'CS?':>5}  Score"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(cross_asset_results, 1):
        m   = r["metrics"]
        siv = f"{m['straddle_avg_iv']:.2f}%"      if m["straddle_avg_iv"] is not None else "  N/A"
        giv = f"{m['garch_vol'] - m['straddle_avg_iv']:+.2f}pp" if m["straddle_avg_iv"] is not None else "  N/A"
        sdl = f"{m['sigma_delta']:.4f}"            if m["sigma_delta"]    is not None else " N/A"
        ads = f"{m['cs_delta_sep']:.4f}"           if m["cs_delta_sep"]   is not None else " N/A"
        ivd = f"{m['cs_iv_diff']:.2f}pp"           if m["cs_iv_diff"]     is not None else " N/A"
        cnc = f"{m['cs_net_credit']:.4f}"          if m["cs_net_credit"]  is not None else "  N/A"
        print(
            f"{i:>3}  {r['asset']:<8}  {siv:>12}  {giv:>9}  "
            f"{sdl:>6}  {ads:>7}  {ivd:>7}  {cnc:>9}  {r['garch_vol']:>7.2f}  "
            f"{'YES' if m['straddle_eligible'] else 'no':>5}  "
            f"{'YES' if m['cs_eligible'] else 'no':>5}  "
            f"{m['skew_score']:.3f}"
        )

    overall = cross_asset_results[0]
    om      = overall["metrics"]
    status  = "ELIGIBLE" if overall["any_eligible"] else "AVAILABLE (below thresholds)"
    print(f"\n{'='*120}")
    print(f"OVERALL BEST SKEW {status}:  {overall['asset']}  "
          f"|  GARCH={overall['garch_vol']:.2f}%  |  spot={overall['spot']:.2f}")
    if om["straddle_eligible"] and om["straddle_call"] and om["straddle_put"]:
        sc = om["straddle_call"]
        sp = om["straddle_put"]
        print(f"  → Cheapest straddle : BUY {sc['option']} (call IV={sc['iv']:.2f}%)  "
              f"+ BUY {sp['option']} (put IV={sp['iv']:.2f}%)  "
              f"K={sc['strike']:.2f}  avg_IV={om['straddle_avg_iv']:.2f}%  "
              f"GARCH−IV={om['garch_vol']-om['straddle_avg_iv']:+.2f}pp")
    if om["cs_eligible"] and om["cs_short"] and om["cs_long"]:
        csh = om["cs_short"]
        clg = om["cs_long"]
        print(f"  → 2σ_Δ call spread  : SELL {csh['option']}  IV={csh['iv']:.2f}%  "
              f"/ BUY {clg['option']}  IV={clg['iv']:.2f}%  "
              f"|IV_diff|={om['cs_iv_diff']:.2f}pp  ΔΔ={om['cs_delta_sep']:.4f}  "
              f"net_credit={om['cs_net_credit']:.4f}")
    print(f"{'='*120}\n")




# ══════════════════════════════════════════════════════════════════════════════
# FLYAGONAL SCANNER
#
# Hybrid defined-risk strategy popularised by Steve Ganz (optionsjive.com):
#
#   CALL BROKEN-WING BUTTERFLY  (above spot, near expiry)
#     BUY  1× lower_call  +  SELL 2× mid_call  +  BUY 1× upper_call
#     Upper wing is intentionally wider than lower wing (broken-wing asymmetry).
#
#   PUT DIAGONAL  (below spot, straddles two expiries)
#     SELL 1× short_put  (near expiry, same rank as BWB)
#     BUY  1× long_put   (same strike, farther expiry)
#
# Combined Greeks at entry
#   Delta : ≈ 0   (structure is placed symmetrically around spot)
#   Theta : strongly positive  (dominant edge)
#   Vega  : ≈ neutral  (BWB slight negative offsets diagonal slight positive)
#
# Entry filter: low GARCH-vol regime; calendar clear of binary events.
# ══════════════════════════════════════════════════════════════════════════════


def _bsm_vega_theta(F, K, T_days, sigma, factor, option_type):
    """
    Compute (vega_per_1pct_iv, daily_theta) for a *long* option position.

    Uses Black-76 forward-price model to stay consistent with the rest of the
    codebase (BlackScholesCalculator uses F = spot/factor as the forward price).

    vega_per_1pct_iv : price change for a +1 percentage-point IV move (positive)
    daily_theta      : price change per 1 trading day (negative = time erodes value)
    """
    if T_days <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0, 0.0
    t    = max(T_days / 252.0, 1e-8)
    d1   = (np.log(F / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2   = d1 - sigma * np.sqrt(t)
    S    = F * factor                                  # spot equivalent

    # Vega: same formula for calls and puts
    vega = S * norm.pdf(d1) * np.sqrt(t) * 0.01       # per 1% IV move

    # Theta: gamma term + interest rate adjustment
    r_eff       = -np.log(max(factor, 1e-10)) / t     # annualised rate from discount factor
    gamma_part  = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(t))
    if option_type == CALL_OPTION:
        r_part = -r_eff * K * factor * norm.cdf(d2)
    else:
        r_part =  r_eff * K * factor * norm.cdf(-d2)
    daily_theta = (gamma_part + r_part) / 252.0

    return vega, daily_theta


def build_flyagonal_candidates(
    utils,
    calls_dict_near,
    puts_dict_near,
    puts_dict_far,
    spot,
    garch_vol,
    T_near,
    factor_near,
    T_far,
    factor_far,
):
    """
    Build and rank all Flyagonal candidates from live option chain data.

    The candidate list is sorted by (eligible, score) descending.
    Each dict contains full leg details plus net Greeks and eligibility flag.
    """
    F_near = spot / max(factor_near, 1e-8)
    F_far  = spot / max(factor_far,  1e-8)

    # ── 1. Call BWB pool ──────────────────────────────────────────────────────
    # De-duplicate near calls by strike, keeping entry closest to Δ 0.30
    call_by_strike = {}
    for delta, data in calls_dict_near.items():
        strike = data["strike"]
        prev   = call_by_strike.get(strike)
        if prev is None or abs(delta - 0.30) < abs(prev[0] - 0.30):
            call_by_strike[strike] = (delta, data)

    if len(call_by_strike) < 3:
        return []

    # Pre-fetch call quotes once per strike
    call_quote = {}
    for strike, (_, data) in call_by_strike.items():
        info = utils.get_option_info_with_quote(data["option_name"])
        if info is not None and info.bid > 0 and info.ask > 0:
            call_quote[strike] = info

    call_strikes = sorted(call_quote.keys())

    # BWB body must be above spot by [BODY_DIST_MIN_PCT, BODY_DIST_MAX_PCT]
    above_spot = [
        s for s in call_strikes
        if FLYAGONAL_BWB_BODY_DIST_MIN_PCT
           <= (s - spot) / max(spot, 1e-8)
           <= FLYAGONAL_BWB_BODY_DIST_MAX_PCT
    ]

    bwb_pool = []
    for mid_strike in above_spot:
        mid_delta, mid_data = call_by_strike[mid_strike]
        if not (FLYAGONAL_BWB_BODY_DELTA_MIN <= mid_delta <= FLYAGONAL_BWB_BODY_DELTA_MAX):
            continue

        for lower_strike in [s for s in call_strikes if s < mid_strike]:
            lower_delta, lower_data = call_by_strike[lower_strike]
            if not (FLYAGONAL_BWB_LOWER_DELTA_MIN <= lower_delta <= FLYAGONAL_BWB_LOWER_DELTA_MAX):
                continue

            lower_wing = mid_strike - lower_strike
            if lower_wing <= 0:
                continue

            for upper_strike in [s for s in call_strikes if s > mid_strike]:
                upper_wing = upper_strike - mid_strike
                # Broken-wing: upper must be wider than lower by the required ratio
                if upper_wing < FLYAGONAL_BWB_BROKEN_WING_RATIO * lower_wing:
                    continue
                # Avoid excessively wide upper wings (capped at 3× lower)
                if upper_wing > 3.0 * lower_wing:
                    continue

                upper_delta, upper_data = call_by_strike[upper_strike]
                lq = call_quote.get(lower_strike)
                mq = call_quote.get(mid_strike)
                uq = call_quote.get(upper_strike)
                if not (lq and mq and uq):
                    continue

                # BWB cost: BUY lower (ask) + SELL 2×mid (bid) + BUY upper (ask)
                bwb_debit  = lq.ask + uq.ask - 2.0 * mq.bid
                max_profit = lower_wing - bwb_debit   # P&L at mid_strike at expiry

                # Net delta of BWB position:  +lower  −2×mid  +upper  (call deltas are positive)
                bwb_delta = lower_delta - 2.0 * mid_delta + upper_delta

                # Greeks per leg (long option convention: theta negative, vega positive)
                iv_l = lower_data["iv"] / 100.0
                iv_m = mid_data["iv"]   / 100.0
                iv_u = upper_data["iv"] / 100.0
                vega_l, th_l = _bsm_vega_theta(F_near, lower_strike, T_near, iv_l, factor_near, CALL_OPTION)
                vega_m, th_m = _bsm_vega_theta(F_near, mid_strike,   T_near, iv_m, factor_near, CALL_OPTION)
                vega_u, th_u = _bsm_vega_theta(F_near, upper_strike, T_near, iv_u, factor_near, CALL_OPTION)

                # Position Greeks:  +lower  −2×mid  +upper
                # theta: long legs earn theta contribution = th_x (negative)
                #        short leg pays theta contribution flipped: -2*th_m (positive if |th_m|>0)
                bwb_theta = th_l + th_u - 2.0 * th_m   # positive when body has highest theta
                bwb_vega  = vega_l + vega_u - 2.0 * vega_m   # negative (short body vega)

                bwb_pool.append({
                    "bwb_debit":     bwb_debit,
                    "max_profit_bwb": max_profit,
                    "bwb_delta":     bwb_delta,
                    "bwb_theta":     bwb_theta,
                    "bwb_vega":      bwb_vega,
                    "body_dist_pct": (mid_strike - spot) / spot,
                    "lower_wing":    lower_wing,
                    "upper_wing":    upper_wing,
                    "body_iv_edge":  garch_vol - mid_data["iv"],
                    "lower_sym":     lower_data["option_name"],
                    "mid_sym":       mid_data["option_name"],
                    "upper_sym":     upper_data["option_name"],
                    "lower_strike":  lower_strike,
                    "mid_strike":    mid_strike,
                    "upper_strike":  upper_strike,
                    "lower_delta":   lower_delta,
                    "mid_delta":     mid_delta,
                    "upper_delta":   upper_delta,
                    "lower_iv":      lower_data["iv"],
                    "mid_iv":        mid_data["iv"],
                    "upper_iv":      upper_data["iv"],
                    "lower_ask":     lq.ask,
                    "mid_bid":       mq.bid,
                    "upper_ask":     uq.ask,
                })

    if not bwb_pool:
        return []

    # ── 2. Put diagonal pool ─────────────────────────────────────────────────
    # Near put eligible: below spot, within dist % range and delta range
    near_put_eligible = {}
    for delta, data in puts_dict_near.items():
        strike   = data["strike"]
        dist_pct = (spot - strike) / max(spot, 1e-8)
        if not (FLYAGONAL_PUT_DIST_MIN_PCT <= dist_pct <= FLYAGONAL_PUT_DIST_MAX_PCT):
            continue
        if not (FLYAGONAL_PUT_DELTA_MIN <= abs(delta) <= FLYAGONAL_PUT_DELTA_MAX):
            continue
        near_put_eligible[strike] = (delta, data)

    # Far put index by strike
    far_put_by_strike = {data["strike"]: (delta, data) for delta, data in puts_dict_far.items()}

    # Pre-fetch put quotes (near eligible + all far)
    put_quote_cache = {}
    for _, (_, data) in near_put_eligible.items():
        name = data["option_name"]
        if name not in put_quote_cache:
            info = utils.get_option_info_with_quote(name)
            if info is not None and info.bid > 0 and info.ask > 0:
                put_quote_cache[name] = info
    for _, (_, data) in far_put_by_strike.items():
        name = data["option_name"]
        if name not in put_quote_cache:
            info = utils.get_option_info_with_quote(name)
            if info is not None and info.bid > 0 and info.ask > 0:
                put_quote_cache[name] = info

    diag_pool = []
    for put_strike, (near_delta, near_data) in near_put_eligible.items():
        short_name = near_data["option_name"]
        sq = put_quote_cache.get(short_name)
        if sq is None or sq.bid <= 0:
            continue

        # Match to far-expiry strike: exact, or nearest within 2%
        if put_strike in far_put_by_strike:
            far_delta, far_data = far_put_by_strike[put_strike]
            used_far_k = put_strike
        else:
            if not far_put_by_strike:
                continue
            nearest = min(far_put_by_strike.keys(), key=lambda k: abs(k - put_strike))
            if abs(nearest - put_strike) / max(put_strike, 1e-8) > 0.02:
                continue
            far_delta, far_data = far_put_by_strike[nearest]
            used_far_k = nearest

        long_name = far_data["option_name"]
        lq = put_quote_cache.get(long_name)
        if lq is None or lq.ask <= 0:
            continue

        # Diagonal cost: BUY far (ask) − SELL near (bid)  → usually small debit
        diag_debit = lq.ask - sq.bid

        iv_near = near_data["iv"] / 100.0
        iv_far  = far_data["iv"]  / 100.0
        vega_near, th_near = _bsm_vega_theta(F_near, put_strike, T_near, iv_near, factor_near, PUT_OPTION)
        vega_far,  th_far  = _bsm_vega_theta(F_far,  used_far_k, T_far,  iv_far,  factor_far,  PUT_OPTION)

        # Position Greeks:  −short_near  +long_far
        # SHORT near put: position theta = −th_near (near decays faster → positive contribution)
        # LONG  far  put: position theta = +th_far  (negative, smaller magnitude)
        diag_theta = -th_near + th_far    # positive (near decays faster)
        diag_vega  = vega_far - vega_near  # positive (longer-dated put has more vega)

        # Net delta of diagonal:  SELL near put (negative delta → position becomes positive)
        #                         BUY  far  put (negative delta → position is negative)
        # near_delta and far_delta are negative; -near_delta turns positive
        diag_delta = -near_delta + far_delta

        diag_pool.append({
            "diag_debit":     diag_debit,
            "diag_delta":     diag_delta,
            "diag_theta":     diag_theta,
            "diag_vega":      diag_vega,
            "put_dist_pct":   (spot - put_strike) / spot,
            "put_strike":     put_strike,
            "far_strike":     used_far_k,
            "near_put_delta": near_delta,
            "far_put_delta":  far_delta,
            "near_put_iv":    near_data["iv"],
            "far_put_iv":     far_data["iv"],
            "short_put_sym":  short_name,
            "long_put_sym":   long_name,
            "short_put_bid":  sq.bid,
            "long_put_ask":   lq.ask,
        })

    if not diag_pool:
        return []

    # ── 3. Combine BWB + diagonal and score ──────────────────────────────────
    candidates = []
    for bwb in bwb_pool:
        for diag in diag_pool:
            raw_debit  = bwb["bwb_debit"] + diag["diag_debit"]
            net_debit  = max(raw_debit, 0.001)          # guard division by zero
            net_delta  = bwb["bwb_delta"] + diag["diag_delta"]
            net_theta  = bwb["bwb_theta"] + diag["diag_theta"]
            net_vega   = bwb["bwb_vega"]  + diag["diag_vega"]

            max_profit  = bwb["max_profit_bwb"]
            reward_risk = max_profit / net_debit if max_profit > 0 else -math.inf

            # Vega-neutrality: how much of the dominant leg's vega is left over
            max_vega_leg = max(abs(bwb["bwb_vega"]), abs(diag["diag_vega"]), 1e-8)
            vega_ratio   = abs(net_vega) / max_vega_leg

            theta_to_debit = net_theta / net_debit

            eligible = (
                raw_debit   > 0
                and raw_debit / spot      <= FLYAGONAL_MAX_NET_DEBIT_PCT
                and net_theta             >  0
                and theta_to_debit        >= FLYAGONAL_MIN_THETA_TO_DEBIT
                and abs(net_delta)        <= FLYAGONAL_MAX_NET_DELTA_ABS
                and vega_ratio            <= FLYAGONAL_MAX_VEGA_RATIO
                and (reward_risk          >= FLYAGONAL_MIN_REWARD_RISK
                     if math.isfinite(reward_risk) else False)
            )

            # Score: theta income, vega neutrality, delta neutrality, reward/risk, low cost
            score = (
                theta_to_debit * 100.0
                + (1.0 - min(vega_ratio, 1.0)) * 20.0
                - abs(net_delta) * 50.0
                + (reward_risk if math.isfinite(reward_risk) else 0.0) * 5.0
                + bwb["body_iv_edge"] * 2.0
                - (raw_debit / spot) * 200.0
            )

            candidates.append({
                "score":          score,
                "eligible":       eligible,
                # Net metrics
                "net_debit":      raw_debit,
                "net_debit_pct":  raw_debit / spot * 100.0,
                "net_delta":      net_delta,
                "net_theta":      net_theta,
                "net_vega":       net_vega,
                "theta_to_debit": theta_to_debit,
                "vega_ratio":     vega_ratio,
                "max_profit":     max_profit,
                "reward_risk":    reward_risk,
                # BWB legs
                "bwb_debit":      bwb["bwb_debit"],
                "bwb_delta":      bwb["bwb_delta"],
                "bwb_theta":      bwb["bwb_theta"],
                "bwb_vega":       bwb["bwb_vega"],
                "body_dist_pct":  bwb["body_dist_pct"],
                "lower_wing":     bwb["lower_wing"],
                "upper_wing":     bwb["upper_wing"],
                "body_iv_edge":   bwb["body_iv_edge"],
                "lower_sym":      bwb["lower_sym"],
                "mid_sym":        bwb["mid_sym"],
                "upper_sym":      bwb["upper_sym"],
                "lower_strike":   bwb["lower_strike"],
                "mid_strike":     bwb["mid_strike"],
                "upper_strike":   bwb["upper_strike"],
                "lower_delta":    bwb["lower_delta"],
                "mid_delta":      bwb["mid_delta"],
                "upper_delta":    bwb["upper_delta"],
                "lower_iv":       bwb["lower_iv"],
                "mid_iv":         bwb["mid_iv"],
                "upper_iv":       bwb["upper_iv"],
                "lower_ask":      bwb["lower_ask"],
                "mid_bid":        bwb["mid_bid"],
                "upper_ask":      bwb["upper_ask"],
                # Diagonal legs
                "diag_debit":     diag["diag_debit"],
                "diag_delta":     diag["diag_delta"],
                "diag_theta":     diag["diag_theta"],
                "diag_vega":      diag["diag_vega"],
                "put_dist_pct":   diag["put_dist_pct"],
                "put_strike":     diag["put_strike"],
                "far_strike":     diag["far_strike"],
                "near_put_delta": diag["near_put_delta"],
                "far_put_delta":  diag["far_put_delta"],
                "near_put_iv":    diag["near_put_iv"],
                "far_put_iv":     diag["far_put_iv"],
                "short_put_sym":  diag["short_put_sym"],
                "long_put_sym":   diag["long_put_sym"],
                "short_put_bid":  diag["short_put_bid"],
                "long_put_ask":   diag["long_put_ask"],
            })

    return sorted(candidates, key=lambda c: (c["eligible"], c["score"]), reverse=True)


async def scan_flyagonal_opportunities(
    asset: str = None,
    bwb_expiry_rank: int = FLYAGONAL_BWB_EXPIRY_RANK,
    diag_long_expiry_rank: int = FLYAGONAL_DIAG_LONG_EXPIRY_RANK,
):
    """
    Scan and rank Flyagonal candidates across all assets (or a single asset).

    For each asset:
      • Loads the near-expiry chain (rank=bwb_expiry_rank) for the BWB + short put.
      • Loads the far-expiry chain  (rank=diag_long_expiry_rank) for the long put.
      • Builds all valid (BWB + diagonal) combinations, scores, and prints results.
      • Appends a cross-asset summary when scanning multiple assets.

    No orders are placed.
    """
    mt5_conn   = MT5Connector()
    quant_calc = QuantCalculation()
    utils      = Utils()

    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    assets_to_scan      = [asset] if asset is not None else ASSET_SYMBOL
    cross_asset_results = []

    for current_asset in assets_to_scan:
        if not mt5_conn.symbol_select(current_asset, True):
            print(f"[{current_asset}] Failed to select — skipping")
            continue

        tick_ok = False
        for _ in range(10):
            tick = mt5_conn.get_mt5_connector().symbol_info_tick(current_asset)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                tick_ok = True
                break
            print(f"Waiting for {current_asset} tick data...")
            await asyncio.sleep(1)
        if not tick_ok:
            print(f"{current_asset} has no tick data after 10 s — skipping")
            continue

        symbol_info = mt5_conn.get_symbol_info(current_asset)
        spot = (symbol_info.bid + symbol_info.ask) / 2

        print(f"\n{'='*96}")
        print(
            f"FLYAGONAL SCANNER  |  {current_asset}  |  spot={spot:.2f}"
            f"  |  BWB rank={bwb_expiry_rank}  |  diag-long rank={diag_long_expiry_rank}"
        )
        print(f"{'='*96}")

        spot_prices_data = mt5_conn.get_data(
            current_asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        regime_tag = ""
        if garch_vol > FLYAGONAL_IV_RANK_PROXY_MAX:
            regime_tag = f"  ← WARNING: {garch_vol:.1f}% > {FLYAGONAL_IV_RANK_PROXY_MAX}% proxy — elevated vol regime"
        print(f"GARCH vol : {garch_vol:.2f}%{regime_tag}")

        # ── Near-expiry chain (BWB + short put) ──────────────────────────────
        chain_near = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=bwb_expiry_rank
        )
        if not chain_near:
            print(f"[{current_asset}] No near-expiry chain (rank {bwb_expiry_rank}) — skipping")
            continue
        expiry_near          = next(iter(chain_near.keys()))
        factor_near, T_near  = utils.get_factor_from_expiration_time(expiry_near)
        print(
            f"Near expiry : {datetime.fromtimestamp(expiry_near).strftime('%Y-%m-%d')}  "
            f"(T={T_near} days,  factor={factor_near:.6f})"
        )

        # ── Far-expiry chain (long put) ───────────────────────────────────────
        chain_far = mt5_conn.get_option_names_by_expiration_time(
            current_asset, expiry_rank_override=diag_long_expiry_rank
        )
        if not chain_far:
            print(f"[{current_asset}] No far-expiry chain (rank {diag_long_expiry_rank}) — skipping")
            continue
        expiry_far          = next(iter(chain_far.keys()))
        factor_far, T_far   = utils.get_factor_from_expiration_time(expiry_far)
        print(
            f"Far  expiry : {datetime.fromtimestamp(expiry_far).strftime('%Y-%m-%d')}  "
            f"(T={T_far} days,  factor={factor_far:.6f})\n"
        )

        if expiry_far <= expiry_near:
            print(
                f"[{current_asset}] Far expiry is not after near expiry "
                f"({datetime.fromtimestamp(expiry_far)} <= {datetime.fromtimestamp(expiry_near)}) — "
                f"try increasing diag_long_expiry_rank."
            )
            continue

        # ── Parse option chains ───────────────────────────────────────────────
        calls_near, puts_near = utils.get_calls_and_puts_data(chain_near, symbol_info)
        _,          puts_far  = utils.get_calls_and_puts_data(chain_far,  symbol_info)

        if not calls_near:
            print(f"[{current_asset}] No calls in near chain")
            continue
        if not puts_near:
            print(f"[{current_asset}] No puts in near chain")
            continue
        if not puts_far:
            print(f"[{current_asset}] No puts in far chain")
            continue

        # ── Print IV surfaces ─────────────────────────────────────────────────
        print(f"  NEAR CALLS  (BWB source):")
        print(f"  {'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}")
        print("  " + "-" * 52)
        for d in sorted(calls_near, reverse=True):
            v = calls_near[d]
            print(f"  {d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}")

        print(f"\n  NEAR PUTS  (short put leg):")
        print(f"  {'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}")
        print("  " + "-" * 52)
        for d in sorted(puts_near):
            v = puts_near[d]
            print(f"  {d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}")

        print(f"\n  FAR PUTS  (long put leg):")
        print(f"  {'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}")
        print("  " + "-" * 52)
        for d in sorted(puts_far):
            v = puts_far[d]
            print(f"  {d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}")

        # ── Build candidates ──────────────────────────────────────────────────
        candidates = build_flyagonal_candidates(
            utils, calls_near, puts_near, puts_far,
            spot, garch_vol, T_near, factor_near, T_far, factor_far,
        )

        if not candidates:
            print(f"\n[{current_asset}] No Flyagonal candidates found with valid live quotes.\n")
            continue

        eligible = [c for c in candidates if c["eligible"]]
        best     = eligible[0] if eligible else candidates[0]

        cross_asset_results.append({
            "asset":           current_asset,
            "garch_vol":       garch_vol,
            "spot":            spot,
            "best":            best,
            "eligible":        bool(eligible),
            "eligible_count":  len(eligible),
            "total":           len(candidates),
            "expiry_near":     expiry_near,
            "expiry_far":      expiry_far,
        })

        # ── Ranked candidate table (top 15) ───────────────────────────────────
        print(f"\n{'='*152}")
        print(
            "RANKED FLYAGONAL CANDIDATES  "
            f"(debit <= {FLYAGONAL_MAX_NET_DEBIT_PCT*100:.1f}%  |  "
            f"θ/debit >= {FLYAGONAL_MIN_THETA_TO_DEBIT:.3f}  |  "
            f"|netΔ| <= {FLYAGONAL_MAX_NET_DELTA_ABS:.2f}  |  "
            f"vega_ratio <= {FLYAGONAL_MAX_VEGA_RATIO:.2f}  |  "
            f"R/R >= {FLYAGONAL_MIN_REWARD_RISK:.2f}  |  "
            f"GARCH={garch_vol:.2f}%)"
        )
        print(f"{'='*168}")
        col = (
            f"{'#':>3}  {'Score':>7}  {'Dbt%':>6}  {'θ/dbt':>7}  {'netΔ':>6}  {'VegaR':>6}  {'R/R':>6}  "
            f"{'BWB_LOW':<14}  {'BWB_MID':<14}  {'BWB_UP':<14}  "
            f"{'PUT_SHORT':<14}  {'PUT_LONG':<14}  OK"
        )
        print(col)
        print("-" * len(col))
        for i, c in enumerate(candidates[:15], 1):
            rr = f"{c['reward_risk']:.2f}" if math.isfinite(c["reward_risk"]) else "N/A"
            ok = "YES" if c["eligible"] else "no"
            print(
                f"{i:>3}  {c['score']:>7.2f}  {c['net_debit_pct']:>6.3f}  "
                f"{c['theta_to_debit']:>7.4f}  {c['net_delta']:>+6.3f}  {c['vega_ratio']:>6.4f}  {rr:>6}  "
                f"{c['lower_sym']:<14}  {c['mid_sym']:<14}  {c['upper_sym']:<14}  "
                f"{c['short_put_sym']:<14}  {c['long_put_sym']:<14}  {ok}"
            )

        # ── Best candidate detail ─────────────────────────────────────────────
        rr_s     = f"{best['reward_risk']:.2f}" if math.isfinite(best["reward_risk"]) else "N/A"
        bwr      = best["upper_wing"] / max(best["lower_wing"], 1e-6)
        label    = "BEST ELIGIBLE" if eligible else "BEST AVAILABLE (none met all thresholds)"
        near_str = datetime.fromtimestamp(expiry_near).strftime('%Y-%m-%d')
        far_str  = datetime.fromtimestamp(expiry_far).strftime('%Y-%m-%d')

        print(f"\n{'='*96}")
        print(f"[{current_asset}] {label} FLYAGONAL:")

        print(f"\n  ── CALL BROKEN-WING BUTTERFLY  (near expiry {near_str}) ──")
        print(
            f"  BUY  1× {best['lower_sym']:<18} K={best['lower_strike']:.2f}"
            f"  Δ={best['lower_delta']:.2f}  IV={best['lower_iv']:.2f}%  ask={best['lower_ask']:.4f}"
        )
        print(
            f"  SELL 2× {best['mid_sym']:<18} K={best['mid_strike']:.2f}"
            f"  Δ={best['mid_delta']:.2f}  IV={best['mid_iv']:.2f}%  bid={best['mid_bid']:.4f}"
        )
        print(
            f"  BUY  1× {best['upper_sym']:<18} K={best['upper_strike']:.2f}"
            f"  Δ={best['upper_delta']:.2f}  IV={best['upper_iv']:.2f}%  ask={best['upper_ask']:.4f}"
        )
        print(
            f"  Lower wing : {best['lower_wing']:.2f}  |  Upper wing : {best['upper_wing']:.2f}"
            f"  |  BW ratio : {bwr:.2f}×  |  Body dist : {best['body_dist_pct']*100:.2f}% above spot"
        )
        print(
            f"  BWB debit  : {best['bwb_debit']:.4f}"
            f"  |  Body IV edge vs GARCH : {best['body_iv_edge']:.2f}pp"
        )

        print(f"\n  ── PUT DIAGONAL  (short {near_str}  /  long {far_str}) ──")
        print(
            f"  SELL 1× {best['short_put_sym']:<18} K={best['put_strike']:.2f}"
            f"  Δ={best['near_put_delta']:.2f}  IV={best['near_put_iv']:.2f}%  bid={best['short_put_bid']:.4f}"
        )
        print(
            f"  BUY  1× {best['long_put_sym']:<18} K={best['far_strike']:.2f}"
            f"  Δ={best['far_put_delta']:.2f}  IV={best['far_put_iv']:.2f}%  ask={best['long_put_ask']:.4f}"
        )
        print(
            f"  Diag debit : {best['diag_debit']:.4f}"
            f"  |  Put dist : {best['put_dist_pct']*100:.2f}% below spot"
        )

        print(f"\n  ── COMBINED FLYAGONAL ──")
        print(
            f"  Net debit      : {best['net_debit']:.4f}  ({best['net_debit_pct']:.3f}% of spot)"
            f"  ← maximum loss"
        )
        print(f"  Approx profit  : {best['max_profit']:.4f}  |  Reward/Risk : {rr_s}")
        print(
            f"  Net delta      : {best['net_delta']:+.4f}"
            f"  ({'≈ neutral' if abs(best['net_delta']) <= FLYAGONAL_MAX_NET_DELTA_ABS else 'WARNING: delta too large'})"
            f"   BWB Δ={best['bwb_delta']:+.4f}  diag Δ={best['diag_delta']:+.4f}"
        )
        print(
            f"  Daily theta    : {best['net_theta']:.6f}"
            f"  |  θ/debit : {best['theta_to_debit']:.4f}"
        )
        print(
            f"  Net vega       : {best['net_vega']:.6f}"
            f"  |  Vega ratio : {best['vega_ratio']:.4f}"
            f"  ({'BWB −vega' if best['bwb_vega'] < 0 else 'BWB +vega'}"
            f" + diag +vega → {'≈ neutral' if best['vega_ratio'] <= 0.30 else 'watch'})"
        )
        print(
            f"  Score : {best['score']:.2f}"
            f"  |  Eligible : {len(eligible)}/{len(candidates)}"
        )
        print(f"{'='*96}\n")

    # ── Cross-asset summary ───────────────────────────────────────────────────
    if len(assets_to_scan) > 1 and cross_asset_results:
        cross_asset_results.sort(
            key=lambda r: (r["eligible"], r["best"]["score"]),
            reverse=True,
        )
        print(f"\n{'='*130}")
        print(
            f"CROSS-ASSET FLYAGONAL SUMMARY  "
            f"({len(cross_asset_results)} of {len(assets_to_scan)} assets returned candidates)"
        )
        print(f"{'='*130}")
        col2 = (
            f"{'#':>3}  {'Asset':<8}  {'GARCH%':>7}  {'Spot':>8}  {'Score':>7}  {'Dbt%':>6}  "
            f"{'θ/dbt':>7}  {'VegaR':>6}  {'R/R':>6}  {'Elig':>4}  "
            f"{'BWB_MID':<18}  {'PUT_DIAG':<18}  OK"
        )
        print(col2)
        print("-" * len(col2))
        for i, r in enumerate(cross_asset_results, 1):
            b  = r["best"]
            rr = f"{b['reward_risk']:.2f}" if math.isfinite(b["reward_risk"]) else "N/A"
            ok = "YES" if r["eligible"] else "no"
            print(
                f"{i:>3}  {r['asset']:<8}  {r['garch_vol']:>7.2f}  {r['spot']:>8.2f}  "
                f"{b['score']:>7.2f}  {b['net_debit_pct']:>6.3f}  "
                f"{b['theta_to_debit']:>7.4f}  {b['vega_ratio']:>6.4f}  {rr:>6}  "
                f"{r['eligible_count']:>4}  "
                f"{b['mid_sym']:<18}  {b['short_put_sym']:<18}  {ok}"
            )

        overall = cross_asset_results[0]
        ob      = overall["best"]
        rr_s    = f"{ob['reward_risk']:.2f}" if math.isfinite(ob["reward_risk"]) else "N/A"
        status  = "ELIGIBLE" if overall["eligible"] else "AVAILABLE (below thresholds)"
        near_s  = datetime.fromtimestamp(overall["expiry_near"]).strftime('%Y-%m-%d')
        far_s   = datetime.fromtimestamp(overall["expiry_far"]).strftime('%Y-%m-%d')
        print(f"\n{'='*130}")
        print(
            f"OVERALL BEST {status}:  {overall['asset']}  |  "
            f"GARCH={overall['garch_vol']:.2f}%  |  spot={overall['spot']:.2f}"
        )
        print(f"  BUY  1× {ob['lower_sym']:<18} K={ob['lower_strike']:.2f}  Δ={ob['lower_delta']:.2f}  IV={ob['lower_iv']:.2f}%  ask={ob['lower_ask']:.4f}")
        print(f"  SELL 2× {ob['mid_sym']:<18} K={ob['mid_strike']:.2f}  Δ={ob['mid_delta']:.2f}  IV={ob['mid_iv']:.2f}%  bid={ob['mid_bid']:.4f}")
        print(f"  BUY  1× {ob['upper_sym']:<18} K={ob['upper_strike']:.2f}  Δ={ob['upper_delta']:.2f}  IV={ob['upper_iv']:.2f}%  ask={ob['upper_ask']:.4f}")
        print(f"  SELL 1× {ob['short_put_sym']:<18} K={ob['put_strike']:.2f}  Δ={ob['near_put_delta']:.2f}  IV={ob['near_put_iv']:.2f}%  bid={ob['short_put_bid']:.4f}  ({near_s})")
        print(f"  BUY  1× {ob['long_put_sym']:<18} K={ob['far_strike']:.2f}  Δ={ob['far_put_delta']:.2f}  IV={ob['far_put_iv']:.2f}%  ask={ob['long_put_ask']:.4f}  ({far_s})")
        print(
            f"  Net debit : {ob['net_debit']:.4f}  ({ob['net_debit_pct']:.3f}%)  |  R/R : {rr_s}"
        )
        print(
            f"  Net delta : {ob['net_delta']:+.4f}"
            f"  ({'≈ neutral' if abs(ob['net_delta']) <= FLYAGONAL_MAX_NET_DELTA_ABS else 'WARNING'})"
            f"  |  Daily θ : {ob['net_theta']:.6f}  |  θ/debit : {ob['theta_to_debit']:.4f}"
            f"  |  Vega ratio : {ob['vega_ratio']:.4f}  |  Score : {ob['score']:.2f}"
        )
        print(f"{'='*130}\n")


asyncio.run(scan_flyagonal_opportunities())
