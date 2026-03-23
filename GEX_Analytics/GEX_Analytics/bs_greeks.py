# -*- coding: utf-8 -*-
"""
Black-Scholes Pricing, Greeks & Implied Volatility
---------------------------------------------------
Pure analytical functions — no external data dependencies.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

RISK_FREE_RATE = 0.1425       # Brazilian SELIC
CONTRACT_MULTIPLIER = 1       # B3 mini options = 1 share


def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes Gamma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes Delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if option_type == 'call' else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1


def implied_vol(price, S, K, T, r, option_type='call'):
    """Solve for implied volatility using Brent's method."""
    if T <= 0 or price <= 0 or K <= 0:
        return 0.30
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if price <= intrinsic + 1e-8:
        return 0.30
    try:
        return brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price,
            0.01, 5.0, xtol=1e-6, maxiter=100
        )
    except (ValueError, RuntimeError):
        return 0.30
