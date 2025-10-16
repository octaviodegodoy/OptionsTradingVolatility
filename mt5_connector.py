import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time
import pandas as pd

from constants import PERIODS, SHIFT_PERIODS

class MT5Connector:
    
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, SHIFT_PERIODS, PERIODS)
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


    def place_order(self,symbolY,symbolX,volumeY,volumeX,orders_type,zscore):

      # prepare the Short request
        #volumeY, volume_X = calculate_volumes(symbolY,symbolX,slope,min_lot_Y,min_lot_X,available_margin,total_positions)
        request_y = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volumeY,
           "type": orders_type[0],
           "zscore": mt5.symbol_info_tick(symbolY).bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": MAGIC_NUMBER,
           "comment": "y,{:.2f}".format(zscore),
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_y_check = mt5.order_check(request_y)
        print("Resultado do short check order (dependente) ", result_y_check)       
        
        # prepare the Long request
        point=mt5.symbol_info(symbolX).point
        request_x = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volumeX,
           "type": orders_type[1],
           "zscore": mt5.symbol_info_tick(symbolX).ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": MAGIC_NUMBER,
           "comment": "x,{:.2f}".format(zscore),
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_x_order_check = mt5.order_check(request_x)
        print("Resultado do long check order (independente) ", result_x_order_check)

        result_y_order = mt5.order_send(request_y)
        result_x_order = mt5.order_send(request_x)
        print("Resultado do short (dependente) ", result_y_order)
        print("Resultado do long (independente) ", result_x_order) 
    
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
    
    def get_options_chain(self,group_name):
        options_symbols = mt5.symbols_get(group_name)
        time_now = int(time.time())
        options_chain = {}
        for s in options_symbols:
            if s.expiration_time > time_now:
               if s.expiration_time not in options_chain:
                   options_chain[s.expiration_time] = []
               options_chain[s.expiration_time].append(s.name)
               options_chain[s.expiration_time].append(s.option_strike)

        sorted_options_chain = dict(sorted(options_chain.items()))
        options_list = list(sorted_options_chain.items())[0]
        return options_list
    
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