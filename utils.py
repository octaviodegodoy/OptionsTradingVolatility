from datetime import datetime, time, timedelta
import logging
from turtle import pd
from functions.black_scholes import BlackScholesCalculator
from constants import ASSET_SYMBOL, BRAZILIAN_HOLIDAYS, CALL_OPTION, PUT_OPTION, STRIKE_PRICE_OFFSET, UNIX_DAYS_IN_SECONDS
import time
import pandas as pd

from mt5_connector import MT5Connector

class Utils:

    def __init__(self):
        self.black_scholes_calculator = BlackScholesCalculator()
        self.mt5_conn = MT5Connector()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def count_weekdays(self, start_date, days_to_add):
        weekdays_count = 0  # Counter for weekdays
        current_date = start_date  # Start counting from this date
        holidays = [datetime.strptime(holiday, "%Y-%m-%d").date() if isinstance(holiday, str) else holiday for holiday in BRAZILIAN_HOLIDAYS]
        for _ in range(days_to_add):
            current_date += timedelta(days=1)  # Move to the next day

            if current_date.weekday() < 5 and current_date.date() not in holidays:  # Check if it's a weekday (0-4 are weekdays)
                weekdays_count += 1

        return weekdays_count

    def get_factor_from_expiration_time(self, expiration_time):
        T = self.get_tenor(expiration_time)
        # get r interest rate
        r = self.black_scholes_calculator.get_interpolated_rate(pd.to_datetime(expiration_time, unit='s'))
        factor = (r/100+1)**((-T)/252)
        return factor,T

    def get_tenor(self, expiration_time):
        time_now = int(time.time())
        total_days = (expiration_time - time_now)/ UNIX_DAYS_IN_SECONDS  
        T = self.count_weekdays(datetime.fromtimestamp(time_now), int(total_days))
        return T
    
    def selection_condition(self, option_info, symbol_info):
        selected_option = self.mt5_conn.symbol_select(option_info.name,True)
        asset_bid = symbol_info.bid
        asset_ask = symbol_info.ask
        asset_market_price = (asset_bid + asset_ask) / 2

        if not selected_option: 
           print(f"Failed to select option {option_info.name}")

        option_bid = option_info.bid
        option_ask = option_info.ask
        K = option_info.option_strike        

        if option_info is None:
            print(f"Failed to get info for option")
            return False
        if option_bid == 0.0 or option_ask == 0.0 or asset_market_price > K * (1 + STRIKE_PRICE_OFFSET) or asset_market_price < K * (1 - STRIKE_PRICE_OFFSET):
            return False
                      
        return True

    def get_option_info_with_quote(self, option_name, retries=2, wait_seconds=0.05):
        """
        Ensure option symbol is selected and retry quote fetch when bid/ask are zero.
        """
        last_info = None
        for attempt in range(retries + 1):
            self.mt5_conn.symbol_select(option_name, True)
            last_info = self.mt5_conn.get_symbol_info(option_name)
            if last_info is not None and last_info.bid > 0.0 and last_info.ask > 0.0:
                return last_info
            if attempt < retries:
                time.sleep(wait_seconds)
        return last_info
    
    def get_calls_and_puts_data(self, chain_options, symbol_info):
        call_deltas_dict = {}
        put_deltas_dict = {}
        factor,T = self.get_factor_from_expiration_time(list(chain_options.keys())[0])
        logging.info(f"Calculating IVs and Deltas for options with T={T} weekdays to expiry. and factor={factor:.6f} ")
        asset_bid = symbol_info.bid
        asset_ask = symbol_info.ask
        self.logger.info(f"Underlying market price: {(asset_bid + asset_ask) / 2:.2f} (bid: {asset_bid:.2f}, ask: {asset_ask:.2f})")
        asset_market_price = (asset_bid + asset_ask) / 2
        F = asset_market_price / factor
        self.logger.info(f"Forward price F : {F:.2f} and underlying market price: {asset_market_price:.2f}")

        options_names_list = chain_options[list(chain_options.keys())[0]]
        
        for option_name in options_names_list:
            selected_option = self.mt5_conn.symbol_select(option_name,True)
            option_info = self.get_option_info_with_quote(option_name)
            
            if not selected_option:
                self.logger.error(f"Failed to select option {option_name}")
                continue
            if option_info is None or option_info.bid == 0.0 or option_info.ask == 0.0:
                self.logger.debug(f"Skipping {option_name}: no valid bid/ask after symbol_select retries")
                continue

            bid_option_price = option_info.bid
            ask_option_price = option_info.ask
            option_market_price = (bid_option_price + ask_option_price)/2
            K = option_info.option_strike
            option_type = option_info.option_right  # 0 for call, 1 for put
            iv = self.black_scholes_calculator.fx_vol(F, K, T, option_market_price, factor, option_type)
            delta = self.black_scholes_calculator.fx_delta(F, K, T, iv, factor, option_type, asset_market_price)
            iv_call_brentq = self.black_scholes_calculator.implied_vol(F, K, T, option_market_price, factor, option_type)
            delta = round(delta, 2)
            iv = iv * 100  # Convert to percentage
            iv_call_brentq = iv_call_brentq * 100  # Convert to percentage
            session_vol = option_info.session_volume if hasattr(option_info, 'session_volume') else 0
            if option_type == CALL_OPTION:
                call_deltas_dict[delta] = {'iv': round(iv, 2), 'option_name': option_name, 'price': option_market_price, 'strike': K, 'session_volume': session_vol}
            elif option_type == PUT_OPTION:
                put_deltas_dict[delta] = {'iv': round(iv, 2), 'option_name': option_name, 'price': option_market_price, 'strike': K, 'session_volume': session_vol}

        return call_deltas_dict, put_deltas_dict
        
    def put_options_count(self):
        count_put = 0
        positions = self.mt5_conn.get_open_positions()
        for pos in positions:
            symbol_info = self.mt5_conn.get_symbol_info(pos.symbol)
            if symbol_info is not None:
                if symbol_info.option_right == PUT_OPTION:
                    count_put += 1
        return count_put

    def call_options_count(self):
        count_call = 0
        positions = self.mt5_conn.get_open_positions()
        for pos in positions:
            symbol_info = self.mt5_conn.get_symbol_info(pos.symbol)
            if symbol_info is not None:
                if symbol_info.option_right == CALL_OPTION:
                    count_call += 1
        return count_call
    
    def get_total_put_deltas(self):
        total_deltas = 0
        positions = self.mt5_conn.get_open_positions()
        print(f"Calculating total put deltas from open positions. Total open positions: {len(positions)}")  
        for pos in positions:
            asset_symbol_info = self.mt5_conn.get_symbol_info(ASSET_SYMBOL[2])
            bid_asset_price = asset_symbol_info.bid
            ask_asset_price = asset_symbol_info.ask
            asset_market_price = (bid_asset_price + ask_asset_price) / 2
            symbol_info = self.mt5_conn.get_symbol_info(pos.symbol)
            expiration_time = symbol_info.expiration_time
            factor,T = self.get_factor_from_expiration_time(expiration_time)
            bid_option_price = symbol_info.bid
            ask_option_price = symbol_info.ask
            option_market_price = (bid_option_price + ask_option_price)/2
            option_type = symbol_info.option_right
            K = symbol_info.option_strike
            F = asset_market_price / factor
            iv = self.black_scholes_calculator.fx_vol(F, K, T, option_market_price, factor, option_type)
            delta = self.black_scholes_calculator.fx_delta(F, K, T, iv, factor, option_type, asset_market_price)
            if pos.type == self.mt5_conn.ORDER_TYPE_SELL and option_type == CALL_OPTION:
                delta = -delta  # Negate delta for short positions
            elif pos.type == self.mt5_conn.ORDER_TYPE_SELL and option_type == PUT_OPTION:
                delta = -delta  # Negate delta for short put positions
            print(f"Position: {pos.symbol}, Type: {'Put' if option_type == PUT_OPTION else 'Call'}, Delta: {delta:.2f}, Volume: {pos.volume} position type {'Sell' if pos.type == self.mt5_conn.ORDER_TYPE_SELL else 'Buy'}")
            total_deltas += delta * pos.volume
        
        return total_deltas
