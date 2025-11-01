import numpy as np
import pandas as pd
from arch import arch_model

from mt5_connector import MT5Connector
mt5_conn = MT5Connector()

df = mt5_conn.get_data("BOVA11")
print(f" Last price {df.tail(1)}")

#df = pd.read_csv("price_data.csv", parse_dates=["Date"])
df = df.sort_values("time").reset_index(drop=True)

# Assuming you have 300 daily prices in a list or loaded from a source
# For example, replace this with your actual data loading
# prices = pd.read_csv('your_prices.csv')['price'].values  # Example if from CSV
prices = df["close"].values  # Use the 'Close' prices from the DataFrame

# Ensure prices are a numpy array
prices = np.array(prices)

# Compute log returns
log_returns = np.log(prices[1:] / prices[:-1])

# Fit AGARCH(1,1) model using GJR-GARCH variant (asymmetric)
# In the arch library, this is specified with vol='Garch', p=1, o=1, q=1
model = arch_model(log_returns, vol='Garch', p=1, o=1, q=1, dist='normal')

# Estimate via maximum likelihood
results = model.fit(disp='off')

# Print the summary
print(results.summary())

# Optional: Extract annualized volatility (assuming 252 trading days)
annualized_vol = results.conditional_volatility[-1] * np.sqrt(252)
print(f"Latest annualized volatility: {annualized_vol}")