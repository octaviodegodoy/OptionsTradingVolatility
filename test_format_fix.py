"""
Test script to verify the format string fix
"""
import pandas as pd
import numpy as np

# Create test data similar to what would be in self.returns
np.random.seed(42)
test_returns = pd.Series(np.random.normal(0.1, 1.5, 100))

print("Testing format string fix...")
print(f"Returns type: {type(test_returns)}")
print(f"Mean type: {type(test_returns.mean())}")
print(f"Std type: {type(test_returns.std())}")

# This would cause the error:
try:
    print(f"Bad format: μ={test_returns.mean():.3f}%, σ={test_returns.std():.3f}%")
except Exception as e:
    print(f"❌ Error with direct formatting: {e}")

# Fixed version:
try:
    mean_val = test_returns.mean().item() if hasattr(test_returns.mean(), 'item') else float(test_returns.mean())
    std_val = test_returns.std().item() if hasattr(test_returns.std(), 'item') else float(test_returns.std())
    print(f"✅ Fixed format: μ={mean_val:.3f}%, σ={std_val:.3f}%")
except Exception as e:
    print(f"❌ Error with fixed formatting: {e}")