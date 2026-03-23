# Options GEX Analytics — Outspoken Market

BOVA11 — B3 Brazilian options via COTAHIST + OI proxy

## What it does

- Global and range-based Put/Call Ratio
- IV skew (OTM puts vs OTM calls)
- Notional by strike (volume financeiro)
- Gamma Exposure (Customer/Dealer)
- Call/Put walls and Gamma Flip
- $IND ↔ BOVA11 Kalman regression & delta-neutral hedge sizing

## Project structure

| File | Purpose |
|---|---|
| `GammaExposureLevels.py` | Main analysis & entry point (`analyze_options` + `main`) |
| `bs_greeks.py` | Black-Scholes pricing, Greeks, implied vol |
| `gex_utils.py` | Gamma flip detection |
| `gex_plots.py` | All matplotlib charts (notional, Friday GEX, all-expiry GEX) |
| `b3_options_loader.py` | B3 COTAHIST data fetch, call/put classification, Greek computation |
| `kalman_price_mapper.py` | Kalman filter $IND ↔ BOVA11 mapping & delta-neutral hedge sizing |
| `constants.py` | Shared constants (`ASSET_SYMBOL`, `PERIODS`, etc.) |
| `mt5_connector.py` | MetaTrader 5 data connector |

## Practical usage — Intraday $IND trading

### Best days to run

| Day | All-expiry GEX reliability | Friday-only GEX | Best use |
|---|---|---|---|
| **Monday** | Most reliable | Plan weekly hedges | Set the week's key levels |
| **Tuesday** | Very reliable | Confirm Monday's levels | Validate / adjust positions |
| **Wednesday** | Good | Weekly gamma building | Mid-week check |
| **Thursday** | Degrading | High gamma, unstable | Watch for pin risk |
| **Friday** | Noisy | Expiration gamma spike | Intraday only |

**Mon/Tue** give the most reliable GEX levels — the full gamma profile is intact after Friday's expiry clears out. By Thursday/Friday, short-dated gamma dominates and walls become unstable; lean on the Friday-specific GEX section instead.

### Recommended intraday timeframe: 15-minute bars

- Dealer hedging rebalances are visible at this granularity.
- Clean wall tests: call wall = resistance, put wall = support.
- Drop to **5-min** if spot is within ±0.5% of the gamma flip (transition zone).

### Session workflow

1. **Pre-market (09:00 BRT):** Run the script → note `$IND` call wall, put wall, gamma flip.
2. **10:00–11:30:** First 6 fifteen-minute bars — price discovery vs GEX levels.
3. **Wall touch on 15-min close** → mean-reversion entry (positive gamma regime: spot below flip).
4. **Wall break on 15-min close** → trend continuation (negative gamma regime: spot above flip).
5. **14:00–16:00:** Strongest dealer hedging flow period; 15-min signals at GEX levels are most reliable.

### Gamma regime quick reference

| Regime | Condition | Dealer behavior | Strategy |
|---|---|---|---|
| Positive gamma | Spot below gamma flip | Dealers dampen moves | Mean-reversion at walls |
| Negative gamma | Spot above gamma flip | Dealers amplify moves | Trend continuation on wall breaks |
| Transition zone | Spot within ±0.5% of flip | Unstable | Reduce size, use 5-min confirmation |
