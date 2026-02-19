import logging
from unittest import result
from scipy import stats
import numpy as np
from arch import arch_model
import pandas as pd
from constants import ANNUAL_TRADING_DAYS
from mt5_connector import MT5Connector
from scipy.optimize import brentq
from scipy.stats import norm
from math import log, sqrt, exp


class QuantCalculation:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mt5_conn = MT5Connector()
        self.logger.info("Quant methods initialized.")


    def d1(self, F, K, T, sigma):
        # d_1 = (Log(Forward / Strike) + Sigma ^ 2 * (Tenor / 252) / 2) / (Sigma * (Tenor / 252) ^ (1 / 2))
        T_years = T / 252
        d1 = (log(F / K) + 0.5 * sigma ** 2 * T_years) / (sigma * sqrt(T_years))
        return d1
    
    def d2(self, F, K, T, sigma):
        d1_value = self.d1(F, K, T, sigma)
        d2 = d1_value - sigma * sqrt(T / 252)
        return d2

    def agarch_estimation(self, prices):
        # Ensure prices are a numpy array
        prices = np.array(prices)

        # Compute log returns
        log_returns = np.log(prices[1:] / prices[:-1])

        # Scale returns by 100 to improve parameter estimation (convert to percentage returns)
        scaled_returns = log_returns * 100
        

        # Fit GARCH(1,1) model
        # Standard GARCH with p=1, q=1
        model = arch_model(scaled_returns, vol='Garch', p=1, o=1, q=1, dist='normal')

        # Estimate via maximum likelihood
        results = model.fit(disp='off')

        # Optional: Extract annualized volatility (assuming 252 trading days)
        annualized_vol = (results.conditional_volatility[-1] / 100) * np.sqrt(ANNUAL_TRADING_DAYS)

        return annualized_vol
    
    def fx_call(self, F, K, T, sigma, r):
        d1_value = self.d1(F, K, T, sigma)
        d2_value = self.d2(F, K, T, sigma)

        Nd1 = norm.cdf(d1_value)
        Nd2 = norm.cdf(d2_value)
        call_price = (F * Nd1 - K * Nd2) * r   # Assuming a risk-free rate of 5%
        
        if call_price < 0:
            call_price = 0.0

        return call_price


    
    def fx_call_vol(self, F, K, T, Price, r):

        max_iter=100
        tolerance=1e-6
        def objective(sigma):
            if sigma <= 0:
                return 1e10  # Penalty for negative
            return self.fx_call(F, K, T, sigma, r) - Price

        try:
            # Brent's method (more robust, no initial guess needed)
            # Search between 0.001% and 500% volatility
            iv = brentq(objective, 0.00001, 5.0, maxiter=max_iter, xtol=tolerance)

            # Validate result
            if iv < 0 or iv > 5:
                return None
                    
            return iv

        except (RuntimeError, ValueError) as e:
            print(f"Warning: IV calculation failed - {str(e)}")
            return None
        

    def are_puts_steeper(delta_puts, put_ivs, delta_calls, call_ivs, min_diff_pp_per_delta=0.0):
        """
        Check if put IVs have a steeper slope than call IVs when fitted linearly.
        
        Args:
            delta_puts: array of put deltas (negative values)
            put_ivs: array of put implied volatilities (%)
            delta_calls: array of call deltas (positive values)
            call_ivs: array of call implied volatilities (%)
            min_diff_pp_per_delta: minimum difference in absolute slope (pp/delta) 
                                   required for puts to be considered steeper.
                                   Default is 0.0 (any difference counts).
        
        Returns:
            bool: True if abs(put_slope) - abs(call_slope) >= min_diff_pp_per_delta,
                  False otherwise
        
        Example:
            # Require puts to be at least 1.0 pp/delta steeper than calls
            result = are_puts_steeper(delta_puts, put_ivs, delta_calls, call_ivs, min_diff_pp_per_delta=1.0)
        
        Note:
            - Put deltas are converted to absolute values for comparison
            - Both fits are done over the same delta range (the put delta range)
            - Steeper means larger absolute slope value (more negative)
        """
        # Handle data length mismatch (drop extra IVs if needed)
    
    
        put_ivs = put_ivs[:len(delta_puts)]
        
        # Convert put deltas to absolute values
        puts_delta = np.abs(delta_puts)
        
        # Determine delta range from puts
        delta_min = puts_delta.min()
        delta_max = puts_delta.max()
        
        # Filter calls to same delta range
        calls_mask = (delta_calls >= delta_min) & (delta_calls <= delta_max)
        calls_delta_filtered = delta_calls[calls_mask]
        calls_iv_filtered = call_ivs[calls_mask]
        
        # Linear fit: IV = intercept + slope * delta
        slope_put, _ = np.polyfit(puts_delta, put_ivs, 1)
        slope_call, _ = np.polyfit(calls_delta_filtered, calls_iv_filtered, 1)
        
        # Calculate difference in absolute slopes
        slope_diff = abs(slope_put) - abs(slope_call)
        
        # Puts are steeper if the difference meets the minimum threshold
        return slope_diff >= min_diff_pp_per_delta

   