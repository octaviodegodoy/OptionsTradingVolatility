"""
Cubic spline and linear interpolation functions.

This module provides interpolation methods for smooth curve fitting:
- c_spline: Cubic spline interpolation using natural boundary conditions
- lin_interp: Linear interpolation between data points
"""

import numpy as np


def c_spline(x_data, y_data, x_eval):
    """
    Cubic spline interpolation with natural boundary conditions.
    
    Given a data set consisting of x values and y values, this function will
    smoothly interpolate a resulting output (y) value from a given input (x) value.
    
    Parameters:
    -----------
    x_data : array-like
        Input x values (independent variable)
    y_data : array-like
        Output y values (dependent variable)
    x_eval : float
        The x value at which to evaluate the spline
        
    Returns:
    --------
    float
        The interpolated y value at x_eval
        
    Raises:
    -------
    ValueError
        If input arrays have different lengths or are empty
    """
    # Convert inputs to numpy arrays
    xin = np.array(x_data, dtype=float)
    yin = np.array(y_data, dtype=float)
    
    # Validate input
    if len(xin) != len(yin):
        raise ValueError("x_data and y_data must have the same length!")
    
    if len(xin) < 2:
        raise ValueError("Need at least 2 data points for interpolation!")
    
    n = len(xin)
    
    # Initialize arrays for second derivatives calculation
    # Using 0-based indexing in Python
    u = np.zeros(n)
    yt = np.zeros(n)  # Second derivatives
    
    # Natural spline boundary condition at start
    yt[0] = 0.0
    u[0] = 0.0
    
    # Forward elimination pass (tridiagonal algorithm)
    for i in range(1, n - 1):
        sig = (xin[i] - xin[i - 1]) / (xin[i + 1] - xin[i - 1])
        p = sig * yt[i - 1] + 2.0
        yt[i] = (sig - 1.0) / p
        
        # Calculate u[i] in two steps for clarity
        u[i] = (yin[i + 1] - yin[i]) / (xin[i + 1] - xin[i]) - \
               (yin[i] - yin[i - 1]) / (xin[i] - xin[i - 1])
        u[i] = (6.0 * u[i] / (xin[i + 1] - xin[i - 1]) - sig * u[i - 1]) / p
    
    # Natural spline boundary condition at end
    qn = 0.0
    un = 0.0
    yt[n - 1] = (un - qn * u[n - 2]) / (qn * yt[n - 2] + 1.0)
    
    # Back-substitution pass
    for k in range(n - 2, -1, -1):
        yt[k] = yt[k] * yt[k + 1] + u[k]
    
    # Now evaluate spline at the given point x_eval
    # Find the correct interval using binary search would be better,
    # but keeping original logic for now
    klo = 0
    khi = n - 1
    
    # Find bracketing interval
    while khi - klo > 1:
        k = (khi + klo) // 2
        if xin[k] > x_eval:
            khi = k
        else:
            klo = k
    
    # Evaluate cubic spline polynomial
    h = xin[khi] - xin[klo]
    
    if h == 0.0:
        raise ValueError("Duplicate x values in input data!")
    
    a = (xin[khi] - x_eval) / h
    b = (x_eval - xin[klo]) / h
    
    y = a * yin[klo] + b * yin[khi] + \
        ((a**3 - a) * yt[klo] + (b**3 - b) * yt[khi]) * (h**2) / 6.0
    
    return y


def lin_interp(x_data, y_data, x_eval):
    """
    Linear interpolation between data points.
    
    Parameters:
    -----------
    x_data : array-like
        Input x values (independent variable), must be sorted
    y_data : array-like
        Output y values (dependent variable)
    x_eval : float
        The x value at which to evaluate the interpolation
        
    Returns:
    --------
    float
        The linearly interpolated y value at x_eval
        
    Raises:
    -------
    ValueError
        If input arrays have different lengths or x_eval is out of bounds
    """
    # Convert inputs to numpy arrays
    xin = np.array(x_data, dtype=float)
    yin = np.array(y_data, dtype=float)
    
    # Validate input
    if len(xin) != len(yin):
        raise ValueError("Vectors' size mismatch!")
    
    if len(xin) < 2:
        raise ValueError("Need at least 2 data points for interpolation!")
    
    # Find the bracketing points
    vert_x1 = None
    vert_y1 = None
    vert_x2 = None
    vert_y2 = None
    
    for i in range(len(xin)):
        if xin[i] <= x_eval:
            vert_x1 = xin[i]
            vert_y1 = yin[i]
            
            # Exact match
            if xin[i] == x_eval:
                return yin[i]
            
            # Set next point if available
            if i + 1 < len(xin):
                vert_x2 = xin[i + 1]
                vert_y2 = yin[i + 1]
        else:
            break
    
    # Check if we found valid bracketing points
    if vert_x1 is None or vert_x2 is None:
        raise ValueError(f"x_eval={x_eval} is outside the range of x_data")
    
    # Linear interpolation formula
    fwd_factor = (vert_y2 - vert_y1) / (vert_x2 - vert_x1)
    result = vert_y1 + fwd_factor * (x_eval - vert_x1)
    
    return result