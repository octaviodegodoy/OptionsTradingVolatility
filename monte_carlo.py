import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def simulate_gbm():

    S0 = 100
    mu = 0.05
    sigma = 0.1
    T = 1
    dt = 1/252
    n_simulations = 252

    n_steps = int(T / dt)
    S = np.zeros((n_simulations, n_steps))
    prices = []
    S[:, 0] = S0
    for t in range(1, n_steps):
        Z = np.random.standard_normal(n_simulations)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S[:, t])

    #plt.plot(prices, color='blue', alpha=0.01)
    #plt.title('Monte Carlo Simulations of Stock Price Paths')
    #plt.xlabel('Time Steps')
    #plt.ylabel('Stock Price')       
    #plt.show()

    return prices

def estimate_future_stock_price():
    n_simulations = 10000
    simulation_results = []

    for _ in range(n_simulations):
        simulated_path = simulate_gbm()
        simulation_results.append(simulated_path[-1])

    prices = simulate_gbm()
    return prices[-1]


def monte_carlo_option_analysis():
    data = np.random.normal(0, 1, 1000)
    mean = np.mean(data)
    std_dev = np.std(data)
    confidence_interval = norm.interval(0.95, loc=mean, scale=std_dev/np.sqrt(len(data)))

    daily_returns = np.exp(data) - 1
    portfolio_variance = np.var(daily_returns)
    portfolio_std_dev = np.sqrt(portfolio_variance)
    print(f"Mean Daily Return: {np.mean(daily_returns)}, Standard Deviation: {np.std(daily_returns)}")
    print(f"Portfolio Variance: {portfolio_variance}")
    print(f"95% Confidence Interval: {confidence_interval}")
    print(f"Mean: {mean}, Standard Deviation: {std_dev}")
    print(f"Portfolio Standard Deviation: {portfolio_std_dev}")



if __name__ == "__main__":
    #monte_carlo_option_analysis()
    #simulate_gbm()
    estimate_future_stock_price()


   