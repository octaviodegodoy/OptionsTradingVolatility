from datetime import datetime, time
import logging
import asyncio
import time
from functions.black_scholes import BlackScholesCalculator
from constants import ASSET_SYMBOL, GARCH_SAMPLE_SIZE, MIN_DAYS_TO_EXPIRY, UNIX_DAYS_IN_SECONDS
from mt5_connector import MT5Connector
from functions.quant_functions import QuantCalculation
from skew_strategy import SkewStrategy
import pandas as pd

from utils import Utils


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    black_scholes_calculator = BlackScholesCalculator()
    utils = Utils()
    
    # get garch volatility
    quant_calc = QuantCalculation()
    spot_prices_data = mt5_conn.get_data(ASSET_SYMBOL[0], mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0)["close"].values
    garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100
    logger.info(f"GARCH Volatility : {garch_vol:.2f}%")


    # get options chain
    chain_options = mt5_conn.get_option_names_by_expiration_time(ASSET_SYMBOL[0])
    logger.info(f"Options Chain for {ASSET_SYMBOL[0]} retrieved {chain_options.values()} options.")
    expiration_time = list(chain_options.keys())[0]
    logger.info(f"Selected Expiration Time: {datetime.fromtimestamp(expiration_time)}")
    options_data = utils.get_calls_and_puts_data(chain_options)
    logger.info(f"Options Data Retrieved: {options_data}")
    #F = utils.get_factor_from_expiration_time(expiration_time)
    #logger.info(f"Tenor Factor from utils function: {F}")
    #T = utils.get_tenor(expiration_time)
    #logger.info(f"Tenor is: {T} weekdays to expiry. and expiration time is {datetime.fromtimestamp(expiration_time)}")
    #r = black_scholes_calculator.get_interpolated_rate(pd.to_datetime(expiration_time, unit='s'))
    #logger.info(f"Interest Rate for {datetime.fromtimestamp(expiration_time).date()}: {r}")
    
   
   


    
    await asyncio.sleep(5)

asyncio.run(main())