import numpy as np
from scipy.stats import norm
from functions.black_scholes import BlackScholesCalculator
from constants import (
    ASSET_SYMBOL, CALL_OPTION, GARCH_SAMPLE_SIZE, PERIODS,
    PUT_SPREAD_EXPIRY_RANK,
    PUT_SPREAD_LONG_DELTA_MIN, PUT_SPREAD_LONG_DELTA_MAX,
    PUT_SPREAD_SHORT_DELTA_MIN, PUT_SPREAD_SHORT_DELTA_MAX,
    PUT_SPREAD_MIN_IV_EDGE, PUT_SPREAD_CALL_WALL_OFFSET,
    SHORT_CALL_BUTTERFLY_EXPIRY_RANK,
    SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT,
    SHORT_CALL_BUTTERFLY_MIN_IV_EDGE,
    SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT,
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
    Scan and rank all bearish put spread candidates for the given asset.
    Prints the full put surface and a sorted table of spread pairs.
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
    atm_price = (symbol_info.bid + symbol_info.ask) / 2

    print(f"\n{'='*72}")
    print(f"PUT SPREAD SCANNER  |  {asset}  |  spot={atm_price:.2f}  |  expiry rank={expiry_rank}")
    print(f"{'='*72}")

    spot_prices_data = mt5_conn.get_data(
        asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0
    )["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
    print(f"GARCH vol : {garch_vol:.2f}%")

    # ── GEX: locate call wall ────────────────────────────────────────────────
    mt5_raw = mt5_conn.get_mt5_connector()
    gex_result = compute_call_wall(mt5_raw, asset, atm_price)
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
        asset, expiry_rank_override=expiry_rank
    )
    if not chain_options:
        print("No option chain returned for selected expiry rank")
        return

    expiration_time = next(iter(chain_options.keys()))
    print(f"Expiry    : {datetime.fromtimestamp(expiration_time)}\n")

    _, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
    if not puts_dict:
        print("No puts returned from chain")
        return

    # ── Full put surface ─────────────────────────────────────────────────────
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

    # ── Build all valid pairs ─────────────────────────────────────────────────
    # When a call wall is available, filter long-leg candidates to those whose
    # strike is closest to call_wall * (1 + PUT_SPREAD_CALL_WALL_OFFSET).
    # We keep the single nearest strike above and below the target.
    long_candidates_all = {
        d: v for d, v in puts_dict.items()
        if PUT_SPREAD_LONG_DELTA_MIN <= abs(d) <= PUT_SPREAD_LONG_DELTA_MAX
    }
    if target_long_strike is not None and long_candidates_all:
        all_long_strikes = sorted({v['strike'] for v in long_candidates_all.values()})
        # Find nearest strike above and below target
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
            iv_edge = garch_vol - l_data['iv']        # positive = long IV cheap vs GARCH
            spread_skew = s_data['iv'] - l_data['iv'] # skew premium collected on short leg
            # Proximity bonus: reward long strikes closer to the call-wall target
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
        print("\nNo valid spread pairs found within configured delta ranges.")
        return

    candidates.sort(key=lambda x: x['score'], reverse=True)

    # ── Ranked table ─────────────────────────────────────────────────────────
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

    # ── Best eligible summary ─────────────────────────────────────────────────
    eligible = [c for c in candidates if c['iv_edge'] >= PUT_SPREAD_MIN_IV_EDGE]
    print(f"\n{'='*80}")
    if eligible:
        b = eligible[0]
        print(f"BEST ELIGIBLE SPREAD:")
        if call_wall:
            print(f"  Call wall          : {call_wall:.2f}  →  target long strike : {target_long_strike:.2f}  (+{PUT_SPREAD_CALL_WALL_OFFSET*100:.0f}%)")
        print(f"  BUY  {b['long_symbol']:<20}  strike={b['long_strike']:.2f}  Δ={b['long_delta']:.2f}  IV={b['long_iv']:.2f}%")
        print(f"  SELL {b['short_symbol']:<20}  strike={b['short_strike']:.2f}  Δ={b['short_delta']:.2f}  IV={b['short_iv']:.2f}%")
        print(f"  IV edge vs GARCH   : {b['iv_edge']:.2f}pp  |  spread skew : {b['spread_skew']:.2f}pp  |  score : {b['score']:.2f}")
    else:
        print(f"No spread meets the minimum IV edge threshold of {PUT_SPREAD_MIN_IV_EDGE}pp vs GARCH ({garch_vol:.2f}%)")
    print(f"{'='*80}\n")


async def scan_short_call_butterfly_opportunities(
    asset: str = None,
    expiry_rank: int = SHORT_CALL_BUTTERFLY_EXPIRY_RANK,
):
    """
    Scan and rank all symmetric short call butterfly candidates for the given asset.
    Prints the call surface and a sorted table using quoted max-profit / max-loss.
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

    print(f"\n{'='*88}")
    print(f"SHORT CALL BUTTERFLY SCANNER  |  {asset}  |  spot={spot:.2f}  |  expiry rank={expiry_rank}")
    print(f"{'='*88}")

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

    calls_dict, _ = utils.get_calls_and_puts_data(chain_options, symbol_info)
    if not calls_dict:
        print("No calls returned from chain")
        return

    print(f"{'Delta':>7} {'Strike':>8} {'IV%':>7}  {'Option':<22}")
    print("-" * 52)
    for d in sorted(calls_dict):
        v = calls_dict[d]
        print(f"{d:>7.2f} {v['strike']:>8.2f} {v['iv']:>7.2f}  {v['option_name']:<22}")

    candidates = build_short_call_butterfly_candidates(utils, calls_dict, spot, garch_vol)
    if not candidates:
        print("\nNo symmetric short call butterfly candidates found with valid live quotes.")
        return

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
    print(f"\n{'='*88}")
    if eligible:
        print("BEST ELIGIBLE SHORT CALL BUTTERFLY:")
    else:
        print("BEST AVAILABLE SHORT CALL BUTTERFLY (none met all thresholds):")
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


asyncio.run(scan_short_call_butterfly_opportunities("PETR4", expiry_rank=2))
