from datetime import datetime, time
import logging
import asyncio
import time
from functions.black_scholes import BlackScholesCalculator
from constants import ASSET_SYMBOL, GARCH_SAMPLE_SIZE, UNIX_DAYS_IN_SECONDS
from mt5_connector import MT5Connector
from functions.quant_functions import QuantCalculation
from skew_strategy import SkewStrategy
import pandas as pd


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    black_scholes_calculator = BlackScholesCalculator()
    
    # get garch volatility
    quant_calc = QuantCalculation()
    spot_prices_data = mt5_conn.get_data(ASSET_SYMBOL[0], mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0)["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100
    logger.info(f"GARCH Volatility : {garch_vol:.2f}%")

    # get options chain
    chain_options = mt5_conn.get_option_names_by_expiration_time(ASSET_SYMBOL[0])
    logger.info(f"Options Chain for {ASSET_SYMBOL[0]} retrieved {chain_options.values()} options.")

    options_names_list = chain_options[list(chain_options.keys())[0]]
    expiration_time = list(chain_options.keys())[0]
    logger.info(f"Listing options from the selected expiration time: {datetime.fromtimestamp(expiration_time)}")

    time_now = int(time.time())
    total_days = (expiration_time - time_now)/ UNIX_DAYS_IN_SECONDS
    logger.info(f"T days from now : {total_days}")
    # get r interest rate
    r = black_scholes_calculator.get_interpolated_rate(pd.to_datetime(expiration_time, unit='s'))
    logger.info(f"Interest Rate for {datetime.fromtimestamp(expiration_time).date()}: {r}")

    
    await asyncio.sleep(5)

asyncio.run(main())