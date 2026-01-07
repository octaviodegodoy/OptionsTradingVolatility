import bisect
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time
import pandas as pd

from constants import CALL_OPTION, MAGIC_NUMBER, MIN_DAYS_TO_EXPIRY, PERIODS, SHIFT_PERIODS

class MT5Connector:

    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1
    
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol, timeframe, periods, shift):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, shift, periods)
        if rates is None:
            self.logger.error(f"Could not get rates for {symbol}")
            return None
        else:
            df = pd.DataFrame(rates)
            df = df.dropna()
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Filter out weekends (keep only weekdays)
            df = df[df['time'].dt.weekday < 5]  # 0=Monday, ..., 4=Friday
            return df   


    def place_order_vertical(self,symbolY,symbolX,orders_type,volume,iv_y,iv_x):
        print(f"Placing vertical order...{symbolY}, {symbolX}, {orders_type}, {volume}, {iv_y}, {iv_x}")

        
        # Enable symbols in Market Watch
        if not mt5.symbol_select(symbolY, True):
            print(f"Failed to select symbol {symbolY}")
            return
        
        if not mt5.symbol_select(symbolX, True):
            print(f"Failed to select symbol {symbolX}")
            return
        
        # Get symbol info
        symbol_info_y = mt5.symbol_info(symbolY)
        symbol_info_x = mt5.symbol_info(symbolX)
        
        if symbol_info_y is None:
            print(f"Symbol {symbolY} not found")
            return
        
        if symbol_info_x is None:
            print(f"Symbol {symbolX} not found")
            return

        print(f"Symbol info X: Ask={symbol_info_x.ask}, Bid={symbol_info_x.bid}, Min={symbol_info_x.volume_min}, Max={symbol_info_x.volume_max}")

        
        # Get the filling mode supported by the symbol
        filling_type_y = symbol_info_y.filling_mode
        filling_type_x = symbol_info_x.filling_mode
        
        # Determine appropriate filling mode for symbol Y
        if filling_type_y & 2:  # FOK is supported
            type_filling_y = mt5.ORDER_FILLING_FOK
        elif filling_type_y & 1:  # IOC is supported
            type_filling_y = mt5.ORDER_FILLING_IOC
        else:  # Return is supported
            type_filling_y = mt5.ORDER_FILLING_RETURN
        
        # Determine appropriate filling mode for symbol X
        if filling_type_x & 1:  # IOC is supported
            type_filling_x = mt5.ORDER_FILLING_IOC
        elif filling_type_x & 4:  # Return is supported
            type_filling_x = mt5.ORDER_FILLING_RETURN
        else:  # FOK is supported
            type_filling_x = mt5.ORDER_FILLING_FOK
        
        print(f"Placing orders for symbols {symbolY} and {symbolX} and order type {orders_type[0]} and order type {orders_type[1]} respectively")

        # Prepare the first request (Y symbol)
        request_y = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volume,
           "type": orders_type[0],
           "price": symbol_info_y.ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": MAGIC_NUMBER,
           "comment": "y,{:.2f}".format(iv_y),
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": type_filling_y,
        }
        
        result_y_check = mt5.order_check(request_y)

        if result_y_check is None:
            print(f"order_check failed for {symbolY}, error: {mt5.last_error()}")
            return
        else:
            print(f"Order check Y result: retcode={result_y_check.retcode}, comment={result_y_check.comment}")
            # retcode 0 means check passed successfully
            if result_y_check.retcode != 0:
                print(f"Order check Y failed: retcode={result_y_check.retcode}, {result_y_check.comment}")
                return

        # Prepare the second request (X symbol)
        request_x = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volume,
           "type": orders_type[1],
           "price": symbol_info_x.bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": MAGIC_NUMBER,
           "comment": "x,{:.2f}".format(iv_x),
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": type_filling_x,
        }
        
        result_x_order_check = mt5.order_check(request_x)

        if result_x_order_check is None:
            print(f"order_check failed for {symbolX}, error: {mt5.last_error()}")
            return
        else:
            print(f"Order check X result: retcode={result_x_order_check.retcode}, comment={result_x_order_check.comment}")
            # retcode 0 means check passed successfully
            if result_x_order_check.retcode != 0:
                print(f"Order check X failed: retcode={result_x_order_check.retcode}, {result_x_order_check.comment}")
                return

        # Both checks passed (retcode == 0), now send orders
        print("Both order checks passed, sending orders...")
        result_y_order = mt5.order_send(request_y)
        result_x_order = mt5.order_send(request_x)

        if result_y_order is None:
            print(f"Order send Y failed, error: {mt5.last_error()}")
        else:
            print(f"Order send Y result: retcode={result_y_order.retcode}, comment={result_y_order.comment}")
            if result_y_order.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Order Y executed successfully! Order: {result_y_order.order}, Deal: {result_y_order.deal}")
            else:
                print(f"Order Y failed: {result_y_order.comment}")

        if result_x_order is None:
            print(f"Order send X failed, error: {mt5.last_error()}")
        else:
            print(f"Order send X result: retcode={result_x_order.retcode}, comment={result_x_order.comment}")
            if result_x_order.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Order X executed successfully! Order: {result_x_order.order}, Deal: {result_x_order.deal}")
            else:
                print(f"Order X failed: {result_x_order.comment}")
    
    def close_all_positions(self):
        # Get all open positions
        positions = mt5.positions_get()
        if positions is not None or len(positions) > 0:
            # Loop through each position and close it
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                volume = position.volume
                position_magic = position.magic
                position_type = position.type  # 0 for buy, 1 for sell

            # Determine the opposite order type to close the position
                if position_type == mt5.ORDER_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    zscore = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    zscore = mt5.symbol_info_tick(symbol).ask

            # Create a close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "position": ticket,
                    "zscore": zscore,
                    "deviation": 20,
                    "magic": MAGIC_NUMBER,
                    "comment": "Close position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

            # Send the close request
                if (position_magic != MAGIC_NUMBER):
                    continue
                result = mt5.order_send(request)

            # Check the result
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to close position {ticket} on {symbol}, Error code: {result.retcode}")
                else:
                    print(f"Successfully closed position {ticket} on {symbol}")

    def get_symbol_futures(self,group_name):
        futures_symbols = mt5.symbols_get(group_name)
        time_now = int(time.time())
        next_symbols_fut = {}
        past_symbols_fut = {}
        for s in futures_symbols:
            if s.expiration_time > time_now:
               next_symbols_fut[s.expiration_time] = s.name
            elif s.expiration_time < time_now:
               past_symbols_fut[s.expiration_time] = s.name
        
        sorted_next_futures = dict(sorted(next_symbols_fut.items()))
        current_symbol = list(sorted_next_futures.items())[0]

        return current_symbol
    
    def get_options_chain(self,group_name,option_type):
        server_info = mt5.account_info().server
        print(f"Connected to MT5 server: {server_info}")
        options_symbols = mt5.symbols_get(group_name)
        print(f"Type for option symbols {type(options_symbols)} options symbols for group {group_name}")
        time_now = int(time.time())
        options_names = []
        expiration_time = 0
        expiration_limit = time_now + MIN_DAYS_TO_EXPIRY #10 days ahead
        # get the last expiration time before expiration_limit
        for s in options_symbols:
            if  s.expiration_time > expiration_limit: #call options only
                expiration_time = s.expiration_time
                break

        for s in options_symbols:
            if s.option_right in [option_type] and s.expiration_time == expiration_time: 
                options_names.append(s.name)

        sorted_options_names = sorted(options_names)
        return sorted_options_names
           
    def get_option_name_by_strike(self,group_name,strike_price,option_type,expiration_time):
        options_symbols = mt5.symbols_get(group_name)
        
        for s in options_symbols:
            
            if s.option_right in [option_type] and s.option_strike == strike_price and s.expiration_time == expiration_time:
               print(f"Found option symbol {s.name} for strike price {strike_price}")
               return s.name
        return None        
    
    def total_daily_risk(self):
        from_date = datetime.now() - timedelta(hours=12,minutes=0)
        #get the number of deals in history
        to_date=datetime.now()
        print(f"From date {from_date} to date {to_date}")
        deals=mt5.history_deals_get(from_date, to_date) 
        total_profit = 0
        total_volume = 0.0
        highest_score = 0.0
        traded_zscore = 0.0
        if deals==None:   
                print("No deals , error code={}".format(mt5.last_error()))   
        elif len(deals) > 0:        
            for deal in deals:
                if (len(deal.comment) > 1):
                    comment_deal = deal.comment.split(",")
                    
                    if (comment_deal[0] == 'y') or (comment_deal[0] == 'x'):
                        traded_zscore = abs(float(comment_deal[1]))
                    if (traded_zscore > highest_score):
                        highest_score = traded_zscore
                total_profit = total_profit + deal.commission + deal.profit
                total_volume = total_volume + deal.volume

        return highest_score,total_profit,total_volume
    
    def get_symbol_info(self,symbol):
        symbol_info = mt5.symbol_info(symbol)
        return symbol_info

    def get_account_info(self):
        account_info = mt5.account_info()
        return account_info    

    def get_open_positions(self):
        positions = mt5.positions_get()
        return positions
    
    def get_total_volume(self):
        total_volume = 0.0
        positions = mt5.positions_get()
        if positions is not None:
            for pos in positions:
                total_volume += pos.volume
        return total_volume
    
    def get_total_positions(self):
        total_positions = mt5.positions_total()
        return total_positions
    
    def last_error():
        last_error = mt5.last_error()
        return last_error

    def sleep(self, seconds):
        time.sleep(seconds)

    def initialize(self):
        return mt5.initialize()

    def shutdown(self):
        mt5.shutdown()

    def get_profit(self):
        profit = mt5.account_info().profit
        return profit
    
    def symbol_select(self,symbol,enable):
        return mt5.symbol_select(symbol,enable)
    
    def get_call_option_name_list(self,group_name):
        server_info = mt5.account_info().server
        print(f"Connected to MT5 server: {server_info}")  
        options_symbols = mt5.symbols_get(group_name)
        time_now = int(time.time())
        options_call_names = {}
        expiration_limit = time_now + MIN_DAYS_TO_EXPIRY #10 days ahead
        print(f"Current time {time_now} and expiration limit {expiration_limit}")
        # get the first expiration time after expiration_limit
        for s in options_symbols:
            
            if s.option_right in [CALL_OPTION] and s.expiration_time > expiration_limit: #call options only
               options_call_names[s.expiration_time] = s.name
               break
        
        sorted_options_call_names = dict(sorted(options_call_names.items()))
        print(f"Sorted call options names: {sorted_options_call_names}")
        return list(sorted_options_call_names.values())
    
    def get_option_names_by_expiration_time(self,symbol):
        
        time_now = int(time.time())
        min_expiration = time_now + MIN_DAYS_TO_EXPIRY #10 days ahead
        expiration_time = 0
        symbol_prefix = symbol[:4]
        group = f"*{symbol_prefix}*,!{symbol}*"
        options_symbols = mt5.symbols_get(group)
        expiration_times = []
        chain_expiration = {}
        
        print(f"Option symbols for group {symbol} size is {len(options_symbols)} ") # meu caralho {options_symbols}")
         # get the first expiration time after min_expiration")
        for s in options_symbols:
            
            if s.expiration_time > min_expiration and s.option_strike > 0:
               expiration_times.append(s.expiration_time)

                       
        sorted_expiration_times = list(dict.fromkeys(expiration_times))
        sorted_expiration_times.sort()

        filtered_names = [symbol.name for symbol in options_symbols if symbol.expiration_time == sorted_expiration_times[0] and symbol.option_strike > 0]

        chain_expiration[sorted_expiration_times[0]] = filtered_names

        return chain_expiration
        
    
    def get_mt5_connector(self):
        return mt5

    def get_put_option_name_list(self,symbol):
        put_option = mt5.symbol_info(symbol)
        return put_option