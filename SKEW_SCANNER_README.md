# Put/Call Skew Opportunity Scanner

This module scans multiple assets for put/call skew opportunities and ranks them by potential profitability.

## Features

- **Multi-Asset Scanning**: Automatically scans a configurable list of assets
- **Skew Analysis**: Calculates linear fit slopes for put and call implied volatilities vs delta
- **GARCH Volatility**: Compares market IV with GARCH forecasted volatility
- **Ranking System**: Ranks opportunities by slope differential magnitude
- **Detailed Reporting**: Provides comprehensive analysis reports

## How It Works

### 1. Skew Calculation

For each asset, the scanner:
1. Retrieves the options chain from MT5
2. Calculates implied volatility (IV) and delta for each option
3. Fits linear regression: `IV = intercept + slope × delta`
4. Compares put slope vs call slope

**Key Metric**: `slope_difference = |put_slope| - |call_slope|`

- **Positive difference** → Puts have steeper skew (bearish tilt)
- **Negative difference** → Calls have steeper skew (bullish tilt)

### 2. Opportunity Identification

The scanner identifies two types of opportunities:

#### LONG_PUT_SKEW
- Put IVs increase more steeply with OTM strikes
- Suggests market fears downside more than usual
- **Strategy**: Buy OTM put spreads, sell ATM volatility

#### LONG_CALL_SKEW
- Call IVs increase more steeply with OTM strikes  
- Suggests market expects strong upside
- **Strategy**: Buy OTM call spreads, sell ATM volatility

### 3. GARCH Comparison

The scanner also calculates GARCH(1,1) forecasted volatility:
- If GARCH vol > Market IV → Volatility underpriced
- If GARCH vol < Market IV → Volatility overpriced

## Usage

### Basic Usage

```python
import asyncio
from main import scan_skew_opportunities, print_opportunity_report

async def run():
    # Scan default assets
    opportunities = await scan_skew_opportunities(
        assets=["PETR4", "VALE3", "ITUB4"],
        days_to_expiry=30,
        min_slope_diff=0.0
    )
    
    # Print formatted report
    print_opportunity_report(opportunities)

asyncio.run(run())
```

### Run from Command Line

```bash
# Configure Python environment first
python main.py
```

### Customization

Edit `main.py` to customize:

```python
# In the main() function:
ASSETS_TO_SCAN = [
    "PETR4",   # Add your assets here
    "VALE3",
    "BBDC4",
    # ... more assets
]

# Adjust scanning parameters
opportunities = await scan_skew_opportunities(
    assets=ASSETS_TO_SCAN,
    days_to_expiry=30,        # Target expiration
    min_slope_diff=1.0        # Minimum slope difference (pp/delta)
)
```

## Output Interpretation

### Sample Report

```
================================================================================
PUT/CALL SKEW OPPORTUNITY REPORT
================================================================================

Found 3 opportunities:

1. PETR4
   Current Price: 34.50
   Opportunity Type: LONG_PUT_SKEW
   Put Slope: -45.235 pp/delta
   Call Slope: -32.145 pp/delta
   Slope Difference: 13.090 pp/delta
   Score: 13.090
   GARCH Vol: 32.50%

2. VALE3
   Current Price: 65.80
   Opportunity Type: LONG_CALL_SKEW
   Put Slope: -28.120 pp/delta
   Call Slope: -41.890 pp/delta
   Slope Difference: -13.770 pp/delta
   Score: 13.770
   GARCH Vol: 28.30%
```

### Understanding the Metrics

- **Put/Call Slope**: Change in IV per unit change in delta (percentage points)
  - More negative = steeper skew curve
  - E.g., -45.235 means IV decreases by ~45 pp when delta increases by 1.0

- **Slope Difference**: Absolute difference between put and call slopes
  - Higher absolute value = stronger opportunity
  - Sign indicates which side has steeper skew

- **Score**: Magnitude of slope difference (used for ranking)

- **GARCH Vol**: Forecasted volatility from GARCH(1,1) model
  - Compare with ATM implied volatility
  - Identifies over/underpriced volatility

## Trading Strategies

### For LONG_PUT_SKEW Opportunities

1. **Put Spread Strategy**
   - Buy OTM put (high IV)
   - Sell further OTM put (even higher IV)
   - Benefit: Sell relatively expensive skew

2. **Put Ratio Spread**
   - Sell 1 ATM put
   - Buy 2-3 OTM puts
   - Benefit: Net credit while maintaining downside protection

3. **Calendar Spread**
   - Sell near-term ATM put
   - Buy longer-term OTM put
   - Benefit: Capture skew and time decay

### For LONG_CALL_SKEW Opportunities

1. **Call Spread Strategy**
   - Buy OTM call (high IV)
   - Sell further OTM call (even higher IV)
   - Benefit: Sell relatively expensive skew

2. **Call Ratio Spread**
   - Sell 1 ATM call
   - Buy 2-3 OTM calls
   - Benefit: Net credit while maintaining upside exposure

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
MetaTrader5>=5.0.4
arch>=5.0.0  # For GARCH models
```

## File Structure

```
main.py                  # Main scanner entry point
skew_scanner.py         # Core skew analysis utilities
mt5_connector.py        # MT5 integration
constants.py            # Configuration constants
functions/
    quant_functions.py  # GARCH and volatility calculations
```

## Configuration

### Constants (constants.py)

```python
ASSET_SYMBOL = "PETR4"              # Default asset
ANNUAL_TRADING_DAYS = 252           # Trading days per year
STRIKE_PRICE_OFFSET = 0.05          # 5% strike range
```

### Risk Parameters

Adjust in `skew_scanner.py`:

```python
# In get_options_chain_with_greeks()
strike_range = 0.20  # ±20% from current price
risk_free_rate = 0.135  # 13.5% annual (adjust for SELIC)
```

## Advanced Analysis

### Additional Metrics

The `skew_scanner.analyze_skew_pattern()` function calculates:

- **ATM Put-Call Spread**: IV difference at 50-delta
- **25-Delta Risk Reversal**: Classic skew indicator
- **Butterfly Spread**: Convexity measure

### Integration with Delta-Neutral Strategies

See `delta_neutral_oportunity.py` for:
- GARCH-based volatility forecasting
- Expected P&L calculations
- Gamma exposure analysis

## Troubleshooting

### No Options Found

- Verify MT5 connection is active
- Check that options are enabled in Market Watch
- Ensure asset symbols are correct
- Verify options expiration dates are available

### IV Calculation Errors

- Check that option prices are valid (bid/ask > 0)
- Ensure strikes are within reasonable range
- Verify days to expiration > 0

### Incorrect Slope Values

- Ensure sufficient data points (≥3 options per side)
- Check delta calculations are correct
- Verify IV values are in decimal format (not %)

## Performance Optimization

For faster scanning:

1. **Reduce Assets**: Scan fewer assets per run
2. **Parallel Processing**: Modify to use `asyncio.gather()`
3. **Cache Data**: Store options data to avoid repeated MT5 calls
4. **Filter Early**: Apply min_slope_diff > 0 to reduce processing

## Examples

### Example 1: Quick Scan

```python
# Scan 3 major assets with minimum threshold
opportunities = await scan_skew_opportunities(
    assets=["PETR4", "VALE3", "ITUB4"],
    days_to_expiry=30,
    min_slope_diff=2.0  # Only strong opportunities
)
```

### Example 2: Comprehensive Analysis

```python
# Scan all liquid assets
LIQUID_ASSETS = [
    "PETR4", "VALE3", "ITUB4", "BBDC4",
    "WEGE3", "B3SA3", "RENT3", "ABEV3"
]

opportunities = await scan_skew_opportunities(
    assets=LIQUID_ASSETS,
    days_to_expiry=45,  # Further out
    min_slope_diff=0.0  # Include all
)

# Export to CSV for further analysis
import pandas as pd
df = pd.DataFrame([vars(opp) for opp in opportunities])
df.to_csv('skew_opportunities.csv', index=False)
```

## References

- Black-Scholes Model for option pricing
- GARCH(1,1) for volatility forecasting
- Linear regression for skew slope calculation
- Risk reversal and butterfly spreads for skew measurement

## License

This is a proprietary trading tool. Not for redistribution.

## Support

For questions or issues:
1. Check MT5 connection status
2. Verify options data availability
3. Review log output for detailed errors
4. Consult `delta_neutral_oportunity.py` for advanced strategies
