import logging
import asyncio
from functions.calculate_garch import GARCHCalculation
from mt5_connector import MT5Connector


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    
    df = mt5_conn.get_data("WEGE3")
    print(f" Last price {df.tail(1)}")
    calculate_garch = GARCHCalculation()
    out = calculate_garch.predict_garch_volatility(df, days_to_expiry=30)
    logger.info(f"Last daily vol (decimal): {out['last_cond_vol_daily']}")
    logger.info(f"5-day period vol (decimal): {out['period_vol_T']}")
    logger.info(f"5-day annualized vol (decimal): {out['annualized_vol_over_T']}")
    logger.info(f"GARCH model parameters: {out['garch_summary']}")
    
    await asyncio.sleep(5)

asyncio.run(main())