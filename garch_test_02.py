import numpy as np
import pandas as pd
from arch import arch_model
import plotly.graph_objects as go

# Assuming you have 300 daily prices in a list or loaded from a source
# For example, replace this with your actual data loading
# prices = pd.read_csv('your_prices.csv')['price'].values  # Example if from CSV
prices = np.random.normal(100, 5, 300)  # Placeholder: 300 random prices around 100 for demonstration

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

# Plot the series using Plotly
# Create a time index (assuming daily data starting from today backwards)
dates = pd.date_range(end=pd.Timestamp.today(), periods=len(log_returns), freq='D')

# Create the plot
fig = go.Figure()

# Add log returns trace
fig.add_trace(go.Scatter(x=dates, y=log_returns, mode='lines', name='Log Returns'))

# Add conditional volatility trace
fig.add_trace(go.Scatter(x=dates, y=results.conditional_volatility, mode='lines', name='Conditional Volatility', yaxis='y2'))

# Update layout for dual y-axis
fig.update_layout(
    title='Log Returns and Conditional Volatility from AGARCH(1,1) Model',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Log Returns'),
    yaxis2=dict(title='Conditional Volatility', overlaying='y', side='right'),
    legend=dict(x=0, y=1)
)

# Show the plot
fig.show()