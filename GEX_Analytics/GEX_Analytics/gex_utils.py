# -*- coding: utf-8 -*-
"""
GEX Utilities — gamma flip detection and related helpers.
"""
import numpy as np
import pandas as pd


def find_gamma_flip(strikes, gex_values, spot, window=3):
    """
    Find the gamma flip point — the zero crossing of smoothed GEX nearest to spot.
    Uses linear interpolation between sign-change points for precision.
    Falls back to NaN if no meaningful crossing exists.
    """
    if len(strikes) < 4:
        return np.nan
    smooth = pd.Series(gex_values).rolling(window, center=True, min_periods=1).mean().values
    # Find all sign-change crossings via linear interpolation
    crossings = []
    for i in range(1, len(smooth)):
        if smooth[i - 1] * smooth[i] < 0:  # sign change
            s0, s1 = strikes[i - 1], strikes[i]
            g0, g1 = smooth[i - 1], smooth[i]
            cross = s0 + (0 - g0) * (s1 - s0) / (g1 - g0)
            crossings.append(cross)
    if crossings:
        # Return the crossing closest to spot
        return min(crossings, key=lambda c: abs(c - spot))
    # Fallback: argmin(|smooth|) but only within ±30% of spot to avoid extreme strikes
    low_bound = spot * 0.70
    high_bound = spot * 1.30
    mask = (strikes >= low_bound) & (strikes <= high_bound)
    if mask.any():
        local_idx = np.where(mask)[0]
        best = local_idx[np.argmin(np.abs(smooth[local_idx]))]
        return strikes[best]
    return np.nan
