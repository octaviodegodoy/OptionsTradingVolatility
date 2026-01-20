PERIODS = 252
SHIFT_PERIODS = 0 
MAGIC_NUMBER = 345346
JUROS_DI_ANUAL = ['F','G','H','J','K','M','N','Q','U','V','X','Z']
CALL_OPTION = 0
PUT_OPTION = 1
UNIX_DAYS_IN_SECONDS = 60*60*24
MIN_DAYS_TO_EXPIRY = 25*UNIX_DAYS_IN_SECONDS # 25 days in seconds
STRIKE_PRICE_OFFSET = 0.05 # 1% above and below current price
TYPE_BUY = 0
TYPE_SELL = 1
OPTION_PRICE_OFFSET = 0.10 # 10% above and below current price
ASSET_SYMBOL = ["BOVA11", "VALE3", "PETR4", "GOAU4", "BBAS3", "BRAV3", "ITUB4", "BBDC4", "MGLU3", "RAIZ4"]
GARCH_SAMPLE_SIZE = 55  # Number of trading days in a year
ANNUAL_TRADING_DAYS = 252
IV_DIFF_THRESHOLD = 0.01  # 5% difference threshold for implied volatility

BRAZILIAN_HOLIDAYS = [
    "2026-02-16",
    "2026-02-17",
    "2026-02-18",
    "2026-04-03",
    "2026-04-21",
    "2026-05-01",
    "2026-06-04",
    "2026-09-07",
    "2026-10-12",
    "2026-11-02",
    "2026-11-20",
    "2026-12-24",
    "2026-12-25",
    "2026-12-31"
]