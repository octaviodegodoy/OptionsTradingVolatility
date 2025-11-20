import logging
from unittest import result
from scipy import stats
import numpy as np
from arch import arch_model
import pandas as pd
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
        print("Got prices for GARCH estimation:", prices)
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

   