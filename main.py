import logging
import asyncio


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting main function...")
        
    await asyncio.sleep(5)

asyncio.run(main())