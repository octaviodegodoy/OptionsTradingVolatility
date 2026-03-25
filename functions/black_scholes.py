import math
from constants import CALL_OPTION, PUT_OPTION
from scipy.stats import norm
from scipy.optimize import newton, brentq
import numpy as np
import pandas as pd
from mt5_connector import MT5Connector
from functions.original_spline import c_spline


class BlackScholesCalculator:
    def __init__(self):
        self.mt5_conn = MT5Connector()

    def d_1(self, F, K, T, sigma):
        return (math.log(F / K) + sigma ** 2 * (T / 252) / 2) / (sigma * (T / 252) ** (1 / 2))
    
    def d_2(self, F, K, T, sigma):
        return self.d_1(F, K, T, sigma) - sigma * (T / 252) ** (1 / 2)
    
    def black_scholes_call(self, F, K, T, sigma, r):
        """European call option price using Black-Scholes forward formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * (T / 252)) / (sigma * np.sqrt(T / 252))
        d2 = d1 - sigma * np.sqrt(T / 252)
        
        # Using forward price formula with discount factor
        call_price = r * (F * norm.cdf(d1) - K * norm.cdf(d2))
        
        return max(call_price, 0.0)

    def black_scholes_put(self, F, K, T, sigma, r):
        """European put option price using Black-Scholes forward formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * (T / 252)) / (sigma * np.sqrt(T / 252))
        d2 = d1 - sigma * np.sqrt(T / 252)
        
        # Using forward price formula with discount factor
        put_price = r * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        return max(put_price, 0.0)

    def implied_vol(self, F, K, T, price, r, option_type):
        """
        Calculate implied volatility using Brent's method.
        
        Args:
            F: Forward price (spot / discount_factor)
            K: Strike price
            T: Time to expiration in trading days
            price: Market price of the option
            r: Discount factor (not the rate itself)
            option_type: 0 for call, 1 for put
        
        Returns:
            Implied volatility as decimal (e.g., 0.20 for 20%)
        """
        if price <= 0 or T <= 0:
            return np.nan
        
        def objective(sigma):
            if sigma <= 0:
                return 1e10
            
            if option_type == CALL_OPTION:  # Call option
                theoretical_price = self.black_scholes_call(F, K, T, sigma, r)
            elif option_type == PUT_OPTION:  # Put option
                theoretical_price = self.black_scholes_put(F, K, T, sigma, r)
            else:
                return 1e10
            
            return theoretical_price - price
        
        try:
            # Search sigma between 0.1% and 500%
            iv = brentq(objective, 0.001, 5.0, maxiter=100)
            return iv
        except (ValueError, RuntimeError) as e:
            # If brentq fails, try with wider bounds or return NaN
            try:
                iv = brentq(objective, 0.0001, 10.0, maxiter=200)
                return iv
            except:
                return np.nan
    

    def fx_delta_call(self, F, K, T, sigma, r, spot):
        x = self.d_1(F, K, T, sigma)
        if spot == 0:
            delta_call = norm.cdf(x)
        else:
            delta_call = F / (spot * r) * norm.cdf(x)
        
        return delta_call
    
    def fx_delta_put(self, F, K, T, sigma, r, spot):
        x = self.d_1(F, K, T, sigma)
        if spot == 0:
            delta_put = norm.cdf(x) - 1
        else:
            delta_put = F / (spot * r) * (norm.cdf(x) - 1)
        
        return delta_put
    
    def fx_delta(self, F, K, T, sigma, r, ID, spot):
        if ID == CALL_OPTION:
            return self.fx_delta_call(F, K, T, sigma, r, spot)
        elif ID == PUT_OPTION:
            return self.fx_delta_put(F, K, T, sigma, r, spot)
    
    def fx_put(self, F, K, T, sigma, r):
        d1_value = self.d_1(F, K, T, sigma)
        d2_value = self.d_2(F, K, T, sigma)

        N_neg_d1 = norm.cdf(-d1_value)
        N_neg_d2 = norm.cdf(-d2_value)
        put_price = (K * N_neg_d2 - F * N_neg_d1) * r  # Assuming a risk-free rate of 5%
        
        if put_price < 0:
            put_price = 0.0

        return put_price
    
    def fx_call(self, F, K, T, sigma, r):
        d1_value = self.d_1(F, K, T, sigma)
        d2_value = self.d_2(F, K, T, sigma)
        N_d1 = norm.cdf(d1_value)
        N_d2 = norm.cdf(d2_value)
        fx_call = (F * N_d1 - K * N_d2) * r

        if fx_call < 0:
            fx_call = 0.0

        return fx_call
    
    def fx_call_vol(self, F, K, T, price, r):
        
        high = 5
        low = 0
        while (high - low) > 0.00000001:
            if self.fx_call(F, K, T, (high + low) / 2, r) > price:
                high = (high + low) / 2
            else:
                low = (high + low) / 2
        
        
        return (high + low) / 2
    
    def fx_put_vol(self, F, K, T, price, r):
        
        high = 5
        low = 0
        while (high - low) > 0.00000001:
            if self.fx_put(F, K, T, (high + low) / 2, r) > price:
                high = (high + low) / 2
            else:
                low = (high + low) / 2
        
        return (high + low) / 2
    
    def fx_vol(self, F, K, T, price, r, ID):
        if ID == CALL_OPTION:
            return self.fx_call_vol(F, K, T, price, r)
        elif ID == PUT_OPTION:
            return self.fx_put_vol(F, K, T, price, r)
        
    def implied_volatility(self, F, K, T, market_price, r, option_type=CALL_OPTION,
                           method='brentq', initial_guess=0.3, max_iter=100, tolerance=1e-6):
        """
        Calculate implied volatility using Newton-Raphson or Brent's method.

        Args:
            F: Forward price
            K: Strike price
            T: Time to expiration in trading days
            market_price: Market price of the option
            r: Discount factor
            option_type: CALL_OPTION or PUT_OPTION
            method: 'newton' or 'brentq'
            initial_guess: Starting sigma for Newton method
            max_iter: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility as decimal, or None if calculation fails.
        """
        if market_price <= 0 or T <= 0:
            return None

        def objective(sigma):
            if sigma <= 0:
                return 1e10
            if option_type == CALL_OPTION:
                return self.black_scholes_call(F, K, T, sigma, r) - market_price
            elif option_type == PUT_OPTION:
                return self.black_scholes_put(F, K, T, sigma, r) - market_price
            return 1e10

        try:
            if method == 'newton':
                iv = newton(objective, initial_guess, maxiter=max_iter, tol=tolerance)
            elif method == 'brentq':
                iv = brentq(objective, 0.001, 5.0, maxiter=max_iter, xtol=tolerance)
            else:
                raise ValueError("method must be 'newton' or 'brentq'")

            if iv < 0 or iv > 5:
                return None
            return iv
        except (RuntimeError, ValueError):
            try:
                iv = brentq(objective, 0.0001, 10.0, maxiter=200)
                if 0 < iv <= 10:
                    return iv
            except (RuntimeError, ValueError):
                pass
            return None

    def fx_gamma(self,F, K, S, T, sigma, r):
        
        x = self.d_1(F, K, T, sigma)
        Pi = math.pi
        a = (1 / (2 * Pi) ** 0.5) * np.exp(-x ** 2 / 2)
        fx_gamma = a * r * F / (S * sigma * (T / 252) ** (1 / 2))
        
        return fx_gamma
    
    def get_interpolated_rate(self, target_date):
        """Get interpolated interest rate for a specific date"""
         # Get interest rate
        data = self.mt5_conn.get_data("DI1@", self.mt5_conn.get_mt5_connector().TIMEFRAME_MN1, 7, 0)
        interest_rates = data.sort_index(ascending=True)

        # Convert timestamps to numeric values (days since first date) for spline interpolation
        time_numeric = (interest_rates['time'] - interest_rates['time'].iloc[0]).dt.total_seconds() / 86400

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        target_numeric = (target_date - interest_rates['time'].iloc[0]).total_seconds() / 86400
        
        rate = c_spline(
            x_data=time_numeric.values,
            y_data=interest_rates['close'].values,
            x_eval=target_numeric
        )
        return rate
