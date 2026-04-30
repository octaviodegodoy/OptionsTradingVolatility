PERIODS = 60
SHIFT_PERIODS = 0 
MAGIC_NUMBER = 345346
CALL_OPTION = 0
PUT_OPTION = 1
UNIX_DAYS_IN_SECONDS = 60*60*24
MIN_DAYS_TO_EXPIRY = 2*UNIX_DAYS_IN_SECONDS # 45 days in seconds
MIN_BIZ_DAYS_TO_EXPIRY = 2  # minimum business days ahead before ranking expirations
TARGET_OPTION_EXPIRY_RANK = 3  # analyze the 3rd upcoming option expiration from now
STRIKE_PRICE_OFFSET = 0.05 # 5% above and below current price
TYPE_BUY = 0
TYPE_SELL = 1
ASSET_SYMBOL = ["BOVA11", "VALE3", "PETR4"] #, "GOAU4", "BBAS3", "BRAV3", "ITUB4", "BBDC4", "MGLU3", "RAIZ4"]
GARCH_SAMPLE_SIZE = 55  # Number of trading days in a year
ANNUAL_TRADING_DAYS = 252
IV_DIFF_THRESHOLD = 0.01  # 5% difference threshold for implied volatility
STEEP_THRESHOLD = 2.0 # Minimum steepness threshold in percentage points per delta
DIFF_IV_GARCH_PUTS_THRESHOLD_PCT = 3 # 1% difference threshold for IV of ATM puts compared to GARCH volatility
MIN_PUT_IV = 15.0 # Minimum IV (%) for ATM put to be considered tradeable
IV_DIFF_THRESHOLD_CALLS = 1.0 # Threshold for IV difference between call strikes to consider for trading
MIN_CALL_SESSION_VOLUME = 0 # Minimum session volume for a call option to be eligible for pair scanning
STRADDLE_MAX_DELTA_IMBALANCE = 0.10  # Target |call_delta + put_delta| for near delta-neutral straddle
STRADDLE_ENTRY_MAX_NET_IV = 22.0  # Entry only if straddle net IV <= this level (%), and this cap is below GARCH
STRADDLE_SLEEP_SECONDS = 25

# ── Strategy constants ───────────────────────────────────────
# Add new strategy names here as you implement them.
VOLATILITY_SKEW = "VOLATILITY_SKEW"
STRADDLE = "STRADDLE"
PUT_SPREAD = "PUT_SPREAD"

ACTIVE_STRATEGY = PUT_SPREAD

# ── Put Spread strategy parameters (bearish / debit spread) ──
PUT_SPREAD_EXPIRY_RANK     = 1      # 1=next expiry, 2=second next, 3=third next
PUT_SPREAD_LONG_DELTA_MIN  = 0.25   # abs(delta) floor for the long put leg  (higher strike, closer ATM)
PUT_SPREAD_LONG_DELTA_MAX  = 0.45   # abs(delta) ceiling for the long put leg
PUT_SPREAD_SHORT_DELTA_MIN = 0.10   # abs(delta) floor for the short put leg (lower strike, further OTM)
PUT_SPREAD_SHORT_DELTA_MAX = 0.24   # abs(delta) ceiling for the short put leg
PUT_SPREAD_MIN_IV_EDGE     = 2.0    # min (garch_vol - long_iv) in pp to enter (long IV cheap vs GARCH)
PUT_SPREAD_MAX_POSITIONS   = 2      # max open put spread sets allowed
PUT_SPREAD_SLEEP_SECONDS   = 25
PUT_SPREAD_CALL_WALL_OFFSET = 0.05  # target long-leg strike = call_wall * (1 + this)

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