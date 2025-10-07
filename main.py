import logging
import asyncio
from functions.calculate_garch import GARCHCalculation


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    garch_calc = GARCHCalculation("BBAS3")
    
    garch_calc.fetch_data()
    garch_calc.calculate_log_returns()
    logger.info("Starting main function...")

    await asyncio.sleep(5)

asyncio.run(main())