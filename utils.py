from datetime import datetime, time, timedelta
import logging
from turtle import pd
from functions.black_scholes import BlackScholesCalculator
from constants import ASSET_SYMBOL, BRAZILIAN_HOLIDAYS, STRIKE_PRICE_OFFSET, UNIX_DAYS_IN_SECONDS
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
        self.logger.info(f"Interest Rate for {datetime.fromtimestamp(expiration_time).date()}: {r}")
        factor = (r/100+1)**((-T)/252)
        self.logger.info(f"Factor : {factor}")
        return factor

    def get_tenor(self, expiration_time):
        time_now = int(time.time())
        total_days = (expiration_time - time_now)/ UNIX_DAYS_IN_SECONDS
        logging.info(f"Total Days to Expiry: {total_days}")    
        T = self.count_weekdays(datetime.fromtimestamp(time_now), int(total_days))
        return T
    
    def selection_condition(self, option_info, symbol_info):
        selected_option = self.mt5_conn.symbol_select(option_info.name,True)
        asset_bid = symbol_info.bid
        asset_ask = symbol_info.ask
        asset_market_price = (asset_bid + asset_ask) / 2

        if not selected_option: 
           print(f"Failed to select option {option_info.name}")

        option_bid_market_price = option_info.bid
        option_ask_market_price = option_info.ask

        option.market_price = 
        K = option_info.option_strike
        if K is not None and ask_market_price > 0.0 and bid_market_price > 0.0:
            logging.info(f"Option Strike Price: {K} tem bid zerado : {bid_market_price} e ask zerado: {ask_market_price} e asset price: {asset_market_price}")

        if option_info is None:
            print(f"Failed to get info for option")
            return False
        if bid_market_price == 0.0 or ask_market_price == 0.0 or asset_market_price > K * (1 + STRIKE_PRICE_OFFSET) or asset_market_price < K * (1 - STRIKE_PRICE_OFFSET):
            return False
        logging.info(f"Option {option_info.name} passed selection condition.")                
        return True
    
    def get_calls_and_puts_data(self, chain_options):
        call_deltas_dict = {}
        put_deltas_dict = {} 

        options_names_list = chain_options[list(chain_options.keys())[0]]
        for option_name in options_names_list:
            option_info = self.mt5_conn.get_symbol_info(option_name)
            selected_option = self.mt5_conn.symbol_select(option_name,True)

            if not selected_option: 
                print(f"Failed to select option {option_name}") 
            else:
                option_info = self.mt5_conn.get_symbol_info(option_name)

                if option_info is None:
                    print(f"Failed to get info for option {option_name}")
                    continue
                if not self.selection_condition(option_info):
                    continue
                K = option_info.option_strike
                logging.info(f"Processing option: {option_name} with strike price: {K}")  


