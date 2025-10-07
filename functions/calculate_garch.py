import logging


class GARCHCalculation:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("GARCH Calculation instance created.")