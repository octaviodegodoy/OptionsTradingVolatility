from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import newton, brentq

class BlackScholesCalculator:
    def __init__(self):
        pass

    def d1(self, F, K, T, sigma):
        # d_1 = (Log(Forward / Strike) + Sigma ^ 2 * (Tenor / 252) / 2) / (Sigma * (Tenor / 252) ^ (1 / 2))
        T_years = T / 252
        d1 = (log(F / K) + 0.5 * sigma ** 2 * T_years) / (sigma * sqrt(T_years))
        return d1
    
    def d2(self, F, K, T, sigma):
        d1_value = self.d1(F, K, T, sigma)
        d2 = d1_value - sigma * sqrt(T / 252)
        return d2
    
    def fx_call(self, F, K, T, sigma, r):
        d1_value = self.d1(F, K, T, sigma)
        d2_value = self.d2(F, K, T, sigma)

        Nd1 = norm.cdf(d1_value)
        Nd2 = norm.cdf(d2_value)
        call_price = (F * Nd1 - K * Nd2) * r   # Assuming a risk-free rate of 5%
        
        if call_price < 0:
            call_price = 0.0

        return call_price
    
    def fx_put(self, F, K, T, sigma, r):
        d1_value = self.d1(F, K, T, sigma)
        d2_value = self.d2(F, K, T, sigma)

        N_neg_d1 = norm.cdf(-d1_value)
        N_neg_d2 = norm.cdf(-d2_value)
        put_price = (K * N_neg_d2 - F * N_neg_d1) * r  # Assuming a risk-free rate of 5%
        
        if put_price < 0:
            put_price = 0.0

        return put_price
    
    def implied_volatility(self, market_price, option_type='call', method='newton', 
                          initial_guess=0.3, max_iter=100, tolerance=1e-6):
        # Select pricing function
        if option_type.lower() == 'call':
            price_func = market_price
        elif option_type.lower() == 'put':
            price_func = market_price
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Objective function: difference between model and market price
        def objective(sigma):
            if sigma <= 0:
                return 1e10  # Penalty for negative volatility
            return price_func(sigma) - market_price
        
        try:
            if method == 'newton':
                # Newton-Raphson method (faster, needs good initial guess)
                iv = newton(objective, initial_guess, maxiter=max_iter, tol=tolerance)
            elif method == 'brentq':
                # Brent's method (more robust, no initial guess needed)
                # Search between 0.001% and 500% volatility
                iv = brentq(objective, 0.00001, 5.0, maxiter=max_iter, xtol=tolerance)
            else:
                raise ValueError("method must be 'newton' or 'brentq'")
            
            # Validate result
            if iv < 0 or iv > 5:
                return None
                
            return iv
        except (RuntimeError, ValueError) as e:
            print(f"Warning: IV calculation failed - {str(e)}")
            return None
    
    def fx_call_vol(self, F, K, T, Price, r,method='bisection'):
        """
        Calculate implied volatility using bisection method
        
        This is a Visual Basic to Python conversion of an implied volatility calculator.
        Uses bisection search to find the volatility that makes fx_call() equal to Price.
        
        Parameters:
        -----------
        Forward : float
            Forward price
        Strike : float
            Strike price
        Tenor : int or float
            Time to expiration in trading days
        Price : float
            Market price of the option
        Interest : float
            Interest/scaling factor
        
        Returns:
        --------
        float
            Implied volatility (sigma)
        
        Example:
        --------
        >>> iv = fx_call_vol(29.44, 28, 26, 1.5, 100000)
        >>> print(f"Implied Volatility: {iv:.8f}")
        """
        try:
            if method == 'bisection':
                high = 5.0
                low = 0.0
                
                while (high - low) > 0.00000001:
                    mid = (high + low) / 2
                    
                    if self.fx_call(F, K, T, mid, r) > Price:
                        high = mid
                    else:
                        low = mid
                
                iv = (high + low) / 2

                 # Validate result
                if iv < 0 or iv > 5:
                    return None
                return iv
        except (RuntimeError, ValueError) as e:
            print(f"Warning: IV calculation failed - {str(e)}")
            return None

if __name__ == "__main__":
    bs_calc = BlackScholesCalculator()
    S = (22.50 + 22.42)/2
    T = 26
    K = 22.69
   
    r = 0.149
    Interest = ((r+1)**(-(T+1)/252))
    print(f"Interest value : {Interest}")
    F = S/Interest
    print(f"Forward value : {F}")
    market_price = (0.45 + 0.95)/2
    print(f"Market Price : {market_price}")    
    sigma = 0.3

    method = 'bisection'

    sigma_call = bs_calc.fx_call_vol(F, K, T, market_price, Interest, method)
    print(f"Implied Volatility (Call) : {sigma_call:.2f}")