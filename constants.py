PERIODS = 60
SHIFT_PERIODS = 0 
MAGIC_NUMBER = 345346
CALL_OPTION = 0
PUT_OPTION = 1
UNIX_DAYS_IN_SECONDS = 60*60*24
MIN_DAYS_TO_EXPIRY = 2*UNIX_DAYS_IN_SECONDS # 45 days in seconds
MIN_BIZ_DAYS_TO_EXPIRY = 2  # minimum business days ahead before ranking expirations
TARGET_OPTION_EXPIRY_RANK = 2  # analyze the 3rd upcoming option expiration from now
STRIKE_PRICE_OFFSET = 0.05 # 5% above and below current price
TYPE_BUY = 0
TYPE_SELL = 1
ASSET_SYMBOL = ["BOVA11", "VALE3", "PETR4", "ITUB4","BBAS3","BBDC4"] #, "GOAU4", "BBAS3", "BRAV3", "ITUB4", "BBDC4", "MGLU3", "RAIZ4"
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
SHORT_CALL_BUTTERFLY = "SHORT_CALL_BUTTERFLY"

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

# ── Short Call Butterfly strategy parameters (short vol wings / long ATM body) ──
SHORT_CALL_BUTTERFLY_EXPIRY_RANK = 3
SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT = 0.03  # body strike must stay within 3% of spot
SHORT_CALL_BUTTERFLY_MIN_IV_EDGE = 1.0  # min (garch_vol - body_iv) in pp
SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT = 0.0  # require structure to be entered for at least flat cashflow
SHORT_CALL_BUTTERFLY_MAX_POSITIONS = 1  # max butterfly sets allowed
SHORT_CALL_BUTTERFLY_ORDER_SIZE = 100.0
SHORT_CALL_BUTTERFLY_SLEEP_SECONDS = 25

# ── Short Strangle strategy parameters ──────────────────────────────────────
SHORT_STRANGLE_EXPIRY_RANK         = 3      # target expiry rank (same logic as others)
SHORT_STRANGLE_CALL_DELTA_MIN      = 0.15   # min call delta for the short call leg (OTM)
SHORT_STRANGLE_CALL_DELTA_MAX      = 0.35   # max call delta for the short call leg
SHORT_STRANGLE_PUT_DELTA_MIN       = 0.15   # min |put delta| for the short put leg (OTM)
SHORT_STRANGLE_PUT_DELTA_MAX       = 0.35   # max |put delta| for the short put leg
SHORT_STRANGLE_MIN_IV_EDGE         = 1.0    # min (leg_iv - garch_vol) in pp required for each leg
SHORT_STRANGLE_MAX_DELTA_IMBALANCE = 0.15   # max |call_delta + put_delta| (portfolio delta neutrality)
SHORT_STRANGLE_MIN_NET_CREDIT      = 0.0    # min combined premium (call_bid + put_bid) to enter

# ── Long Call Butterfly strategy parameters ──────────────────────────────────
# Structure: BUY lower call + SELL 2x middle call + BUY upper call  (net debit)
# Profit if underlying finishes near the middle (body) strike at expiration.
# Enter when body IV is rich vs GARCH (good to sell) and wings are cheap (good to buy).
LONG_CALL_BUTTERFLY_EXPIRY_RANK           = 3      # target expiry rank
LONG_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT = 0.03   # body strike must stay within 3% of spot
LONG_CALL_BUTTERFLY_MIN_BODY_RICHNESS     = 1.0    # min (middle_iv - garch_vol) in pp to enter
LONG_CALL_BUTTERFLY_MIN_REWARD_RISK       = 1.0    # min (max_profit / net_debit) required

# ── Iron Condor strategy parameters ─────────────────────────────────────────
# Structure: BUY lp + SELL sp + SELL sc + BUY lc  (net credit)
#   lp_strike < sp_strike < spot < sc_strike < lc_strike
# Profit if spot stays between the two short strikes at expiration.
# Scored on probability of profit (delta approx) and reward/risk ratio.
IRON_CONDOR_EXPIRY_RANK         = 2     # target expiry rank
IRON_CONDOR_SHORT_DELTA_MIN     = 0.15  # min delta for each short leg (OTM boundary)
IRON_CONDOR_SHORT_DELTA_MAX     = 0.35  # max delta for each short leg
IRON_CONDOR_MIN_WING_WIDTH      = 1.0   # minimum width (price units) per spread side
IRON_CONDOR_MIN_IV_EDGE         = 1.0   # min (short_leg_iv - garch_vol) in pp per short leg
IRON_CONDOR_MIN_POP             = 0.60  # minimum probability of profit (delta approximation)
IRON_CONDOR_MIN_REWARD_RISK     = 0.25  # minimum net_credit / max_loss
IRON_CONDOR_MAX_DELTA_IMBALANCE = 0.15  # max |sc_delta + sp_delta| for near-neutrality

# ── Volatility Skew strategy parameters ─────────────────────────────────────
# Three trades are evaluated per asset:
#   1. Risk Reversal  : SELL ~RR_DELTA put  +  BUY ~RR_DELTA call
#   2. Skew Put Spread: BUY ~NEAR_DELTA put +  SELL ~FAR_DELTA put  (exploit steep put skew)
#   3. Skew Call Spread: SELL ~NEAR_DELTA call + BUY ~FAR_DELTA call (bear call spread for credit)
SKEW_EXPIRY_RANK        = 2     # target expiry rank
SKEW_RR_DELTA           = 0.25  # delta point for risk reversal measurement (25Δ)
SKEW_NEAR_DELTA         = 0.35  # "nearer OTM" delta leg for slope / spread
SKEW_FAR_DELTA          = 0.15  # "farther OTM" delta leg for slope / spread
SKEW_DELTA_TOLERANCE    = 0.08  # max |found_delta − target_delta| to accept a match
SKEW_MIN_RR_PP          = 1.0   # min (put_iv − call_iv) at RR_DELTA in pp to flag skew
SKEW_MIN_SLOPE_PP_PER_DELTA = 2.0  # min pp/Δ slope on the put side to flag steep skew

# ── Flyagonal strategy parameters ────────────────────────────────────────────
# Hybrid: Call Broken-Wing Butterfly (above spot, near expiry)
#       + Put Diagonal (below spot; SELL near put, BUY same-strike far put)
#
# Reference: optionsjive.com/blog/flyagonal-options-strategy/ (Steve Ganz)
#
# Greeks target at entry:
#   Delta  ≈ 0   (natural from structure placement)
#   Theta  > 0   (strongly positive — the primary edge)
#   Vega   ≈ 0   (BWB slight negative offsets diagonal slight positive)
FLYAGONAL_BWB_EXPIRY_RANK        = 2      # expiry rank for BWB legs + short put
FLYAGONAL_DIAG_LONG_EXPIRY_RANK  = 3      # expiry rank for the long put in the diagonal
FLYAGONAL_BWB_BODY_DELTA_MIN     = 0.15   # min Δ for BWB short (body) calls
FLYAGONAL_BWB_BODY_DELTA_MAX     = 0.35   # max Δ for BWB short (body) calls
FLYAGONAL_BWB_LOWER_DELTA_MIN    = 0.20   # min Δ for BWB long lower call
FLYAGONAL_BWB_LOWER_DELTA_MAX    = 0.55   # max Δ for BWB long lower call
FLYAGONAL_BWB_BODY_DIST_MIN_PCT  = 0.01   # body strike must be ≥ 1% above spot
FLYAGONAL_BWB_BODY_DIST_MAX_PCT  = 0.08   # body strike must be ≤ 8% above spot
FLYAGONAL_BWB_BROKEN_WING_RATIO  = 1.3    # upper_wing / lower_wing ≥ this ratio
FLYAGONAL_PUT_DELTA_MIN          = 0.08   # min abs(Δ) for the put diagonal strike
FLYAGONAL_PUT_DELTA_MAX          = 0.25   # max abs(Δ) for the put diagonal strike
FLYAGONAL_PUT_DIST_MIN_PCT       = 0.01   # put strike must be ≥ 1% below spot
FLYAGONAL_PUT_DIST_MAX_PCT       = 0.08   # put strike must be ≤ 8% below spot
FLYAGONAL_MAX_NET_DEBIT_PCT      = 0.05   # max total net debit as % of spot (5%)
FLYAGONAL_MIN_THETA_TO_DEBIT     = 0.005  # min (daily_theta / net_debit)
FLYAGONAL_MAX_VEGA_RATIO         = 0.60   # max |net_vega| / larger_leg_vega
FLYAGONAL_MAX_NET_DELTA_ABS      = 0.10   # max |net_delta| of the full structure (delta-neutral)
FLYAGONAL_MIN_REWARD_RISK        = 0.30   # min (approx_max_profit / net_debit)
FLYAGONAL_IV_RANK_PROXY_MAX      = 30.0   # warn when GARCH vol > this % (high-IV proxy)

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

