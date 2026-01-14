"""
Skew Scanner Module
Provides utilities for scanning and analyzing put/call skew opportunities
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

from mt5_connector import MT5Connector
from constants import CALL_OPTION, PUT_OPTION


def get_options_chain_with_greeks(
    mt5_conn: MT5Connector,
    asset_symbol: str,
    min_expiration_days: int = 25,
    max_expiration_days: int = 35,
    strike_range: float = 0.20
) -> Tuple[List[dict], List[dict]]:
    """
    Get options chain with calculated Greeks for puts and calls
    
    Args:
        mt5_conn: MT5 connector instance
        asset_symbol: Underlying asset symbol
        min_expiration_days: Minimum days to expiration (business days)
        max_expiration_days: Maximum days to expiration (business days)
        strike_range: Strike price range as % from current price (e.g., 0.20 = ±20%)
    
    Returns:
        Tuple of (put_options, call_options) where each is a list of dicts with:
        - symbol: option symbol
        - strike: strike price
        - mid_price: mid market price
        - iv: implied volatility
        - delta: option delta
        - days_to_expiry: calendar days to expiration
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get current underlying price
        df = mt5_conn.get_data(asset_symbol, mt5_conn.TIMEFRAME_D1, 1, 0)
        if df is None or df.empty:
            logger.error(f"Cannot get current price for {asset_symbol}")
            return [], []
        
        current_price = float(df['close'].iloc[-1])
        
        # Get options by expiration
        options_by_expiry = mt5_conn.get_option_names_by_expiration_time(asset_symbol)
        
        if not options_by_expiry:
            logger.warning(f"No options found for {asset_symbol}")
            return [], []
        
        # Convert business days to calendar days (approximate)
        # Business days * 7/5 = calendar days (assuming 5 trading days per week)
        min_calendar_days = int(min_expiration_days * 7 / 5)
        max_calendar_days = int(max_expiration_days * 7 / 5)
        
        # Find all expirations within the range
        valid_expiries = []
        
        for expiry_timestamp, options in options_by_expiry.items():
            expiry_date = datetime.fromtimestamp(expiry_timestamp)
            days_diff = (expiry_date - datetime.now()).days
            
            # Check if expiration is within the desired range
            if min_calendar_days <= days_diff <= max_calendar_days:
                valid_expiries.append((expiry_timestamp, days_diff, options))
                logger.info(f"Found expiration in {days_diff} calendar days (~{int(days_diff * 5/7)} business days)")
        
        if not valid_expiries:
            logger.warning(f"No expirations found between {min_expiration_days}-{max_expiration_days} business days for {asset_symbol}")
            return [], []
        
        # Aggregate options from all valid expirations
        all_options_list = []
        for expiry_timestamp, days_diff, options in valid_expiries:
            for opt in options:
                all_options_list.append((opt, days_diff))
        
        logger.info(f"Total options to process: {len(all_options_list)}")
        
        # Define strike range
        strike_min = current_price * (1 - strike_range)
        strike_max = current_price * (1 + strike_range)
        
        put_options = []
        call_options = []
        
        # Process each option
        for option_symbol, days_to_expiry in all_options_list:
            
            try:
                # Get option info from MT5
                option_info = mt5_conn.get_symbol_info(option_symbol)
                
                if option_info is None:
                    continue
                
                # Extract strike from symbol name
                selected_option = mt5_conn.symbol_select(option_symbol,True)

                if not selected_option: 
                    print(f"Failed to select option {option_symbol}") 
                
                strike = option_info.option_strike
                logger.info(f"Processing symbol {option_symbol} strike {strike} days to expiry: {days_to_expiry}")

                
                if strike is None or strike < strike_min or strike > strike_max:
                    continue
                
                # Get market prices
                bid = option_info.bid
                ask = option_info.ask
                logger.info(f"Option {option_symbol} bid: {bid}, ask: {ask}")
                
                if bid is None or ask is None or bid <= 0 or ask <= 0:
                    continue
                
                mid_price = (bid + ask) / 2
                
                # Determine option type from symbol
                is_call = is_call_option(option_symbol)
                
                # Calculate IV and Greeks
                iv, delta = calculate_iv_and_delta(
                    current_price, strike, mid_price, 
                    days_to_expiry, is_call
                )
                
                if iv is None or delta is None:
                    continue
                
                option_data = {
                    'symbol': option_symbol,
                    'strike': strike,
                    'mid_price': mid_price,
                    'bid': bid,
                    'ask': ask,
                    'iv': iv,
                    'delta': delta,
                    'days_to_expiry': days_to_expiry
                }
                
                if is_call:
                    call_options.append(option_data)
                else:
                    put_options.append(option_data)
                    
            except Exception as e:
                logger.debug(f"Error processing option {option_symbol}: {e}")
                continue
        
        # Sort by strike
        put_options.sort(key=lambda x: x['strike'])
        call_options.sort(key=lambda x: x['strike'])
        
        logger.info(f"Found {len(put_options)} puts and {len(call_options)} calls for {asset_symbol}")
        
        return put_options, call_options
        
    except Exception as e:
        logger.error(f"Error getting options chain for {asset_symbol}: {e}")
        return [], []


def extract_strike_from_symbol(option_info):
    """
    Extract strike price from option symbol
    Example: "PETR4A123" -> extract 123 as strike
    
    This is a placeholder - implement based on your broker's symbol format
    """
    # TODO: Implement based on actual MT5 symbol naming convention
    # This is a simplified example
    try:
       return option_info.strike
    except:
        pass
    return None


def is_call_option(option_symbol: str) -> bool:
    """
    Determine if option is a call based on symbol
    
    This is a placeholder - implement based on your broker's symbol format
    """
    # TODO: Implement based on actual MT5 symbol naming convention
    # Common convention: certain letters indicate calls vs puts
    # For B3 options: A-L are calls, M-X are puts
    month_letters_calls = 'ABCDEFGHIJKL'
    month_letters_puts = 'MNOPQRSTUVWX'
    
    for char in option_symbol:
        if char in month_letters_calls:
            return True
        elif char in month_letters_puts:
            return False
    
    return True  # Default to call if uncertain


def calculate_iv_and_delta(
    spot: float,
    strike: float,
    option_price: float,
    days_to_expiry: int,
    is_call: bool,
    risk_free_rate: float = 0.135  # ~13.5% annual rate (adjust for current SELIC)
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate implied volatility and delta using Black-Scholes
    
    Args:
        spot: Current underlying price
        strike: Strike price
        option_price: Market price of option
        days_to_expiry: Days to expiration
        is_call: True for call, False for put
        risk_free_rate: Annual risk-free rate (decimal)
    
    Returns:
        Tuple of (implied_volatility, delta) or (None, None) if calculation fails
    """
    from scipy.optimize import brentq
    from scipy.stats import norm
    
    try:
        T = days_to_expiry / 252.0  # Convert to years (trading days)
        
        if T <= 0 or spot <= 0 or strike <= 0 or option_price <= 0:
            return None, None
        
        # Black-Scholes formulas
        def bs_price(sigma):
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if is_call:
                return spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
            else:
                return strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        # Calculate implied volatility using Brent's method
        try:
            iv = brentq(lambda sigma: bs_price(sigma) - option_price, 0.01, 5.0)
        except:
            return None, None
        
        # Calculate delta
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        return iv, delta
        
    except Exception as e:
        return None, None


def analyze_skew_pattern(
    put_options: List[dict],
    call_options: List[dict],
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Analyze skew patterns from options chain
    
    Returns:
        Dictionary with skew analysis including:
        - put_skew_slope
        - call_skew_slope
        - skew_differential
        - atm_put_call_spread
        - risk_reversal_25delta
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    analysis = {
        'put_skew_slope': None,
        'call_skew_slope': None,
        'skew_differential': None,
        'atm_put_call_spread': None,
        'risk_reversal_25delta': None,
        'butterfly_spread': None
    }
    
    try:
        # Extract IVs and deltas
        if len(put_options) >= 3:
            put_deltas = np.abs([opt['delta'] for opt in put_options])
            put_ivs = np.array([opt['iv'] for opt in put_options])
            
            # Linear fit for put skew
            put_slope, _ = np.polyfit(put_deltas, put_ivs, 1)
            analysis['put_skew_slope'] = float(put_slope)
        
        if len(call_options) >= 3:
            call_deltas = np.array([opt['delta'] for opt in call_options])
            call_ivs = np.array([opt['iv'] for opt in call_options])
            
            # Linear fit for call skew
            call_slope, _ = np.polyfit(call_deltas, call_ivs, 1)
            analysis['call_skew_slope'] = float(call_slope)
        
        # Calculate skew differential
        if analysis['put_skew_slope'] is not None and analysis['call_skew_slope'] is not None:
            analysis['skew_differential'] = abs(analysis['put_skew_slope']) - abs(analysis['call_skew_slope'])
        
        # ATM put-call IV spread
        if put_options and call_options:
            # Find near-ATM options (delta closest to 0.5)
            atm_put = min(put_options, key=lambda x: abs(abs(x['delta']) - 0.5))
            atm_call = min(call_options, key=lambda x: abs(x['delta'] - 0.5))
            analysis['atm_put_call_spread'] = atm_put['iv'] - atm_call['iv']
        
        # 25-delta risk reversal (25-delta call IV - 25-delta put IV)
        put_25d = [opt for opt in put_options if 0.20 < abs(opt['delta']) < 0.30]
        call_25d = [opt for opt in call_options if 0.20 < opt['delta'] < 0.30]
        
        if put_25d and call_25d:
            analysis['risk_reversal_25delta'] = call_25d[0]['iv'] - put_25d[0]['iv']
        
        # Butterfly spread (put wing + call wing - 2*ATM)
        if len(put_options) >= 2 and len(call_options) >= 2:
            otm_put = put_options[0]  # Lowest strike
            otm_call = call_options[-1]  # Highest strike
            
            # Find ATM
            all_strikes = [opt['strike'] for opt in put_options + call_options]
            mid_strike = np.median(all_strikes)
            
            atm_option = min(put_options + call_options, 
                           key=lambda x: abs(x['strike'] - mid_strike))
            
            analysis['butterfly_spread'] = (otm_put['iv'] + otm_call['iv']) / 2 - atm_option['iv']
        
    except Exception as e:
        logger.error(f"Error analyzing skew pattern: {e}")
    
    return analysis


def format_skew_report(asset: str, analysis: dict) -> str:
    """Format skew analysis into readable report"""
    report = f"\n{'='*60}\n"
    report += f"SKEW ANALYSIS: {asset}\n"
    report += f"{'='*60}\n"
    
    if analysis['put_skew_slope'] is not None:
        report += f"Put Skew Slope: {analysis['put_skew_slope']:.3f} pp/delta\n"
    
    if analysis['call_skew_slope'] is not None:
        report += f"Call Skew Slope: {analysis['call_skew_slope']:.3f} pp/delta\n"
    
    if analysis['skew_differential'] is not None:
        report += f"Skew Differential: {analysis['skew_differential']:.3f} pp/delta\n"
        if analysis['skew_differential'] > 0:
            report += "→ Puts have steeper skew (bearish tilt)\n"
        else:
            report += "→ Calls have steeper skew (bullish tilt)\n"
    
    if analysis['atm_put_call_spread'] is not None:
        report += f"ATM Put-Call IV Spread: {analysis['atm_put_call_spread']:.2%}\n"
    
    if analysis['risk_reversal_25delta'] is not None:
        report += f"25-Delta Risk Reversal: {analysis['risk_reversal_25delta']:.2%}\n"
    
    if analysis['butterfly_spread'] is not None:
        report += f"Butterfly Spread: {analysis['butterfly_spread']:.2%}\n"
    
    report += f"{'='*60}\n"
    
    return report
