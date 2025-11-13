from math import log, sqrt, exp

class BlackScholesCalculator:
    def __init__(self):
        pass

    def d1(self, F, K, T, sigma):
        log_value = log(F / K)
        print(f"log_value: {log_value}")
        sigma_squared_term = (sigma ** 2) * (T / 252) / 2
        print(f"sigma_squared_term: {sigma_squared_term}")
        d1 = (log_value + sigma_squared_term) / (sigma * sqrt(T / 252))
        print(f"d1: {d1}")
        return d1
    
    def d2(self, F, K, T, sigma):
        d1_value = self.d1(F, K, T, sigma)
        d2 = d1_value - sigma * sqrt(T / 252)
        print(f"d2: {d2}")
        return d2

if __name__ == "__main__":
    bs_calc = BlackScholesCalculator()
    F = 32.95
    K = 32.5
    T = 26
    sigma = 0.3

    d1 = bs_calc.d1(F, K, T, sigma)
    print(f"d1 value : {d1}")
    d2 = bs_calc.d2(F, K, T, sigma)
    print(f"d2 value : {d2}")
