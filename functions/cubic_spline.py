"""
c_spline.py

Natural cubic spline interpolation ported from the provided VBA routine.

Function:
    c_spline(xi, yi, x_query)

- xi: sequence of x sample points (1D array-like)
- yi: sequence of y sample points (1D array-like), same length as xi
- x_query: scalar or 1D array-like of x values where you want interpolated y

Returns:
- If x_query is scalar -> float
- If x_query is array-like -> numpy.ndarray of interpolated values

Notes:
- This implements a natural cubic spline (second derivatives are zero at endpoints),
  following the algorithm structure in the original VBA (Numerical Recipes style).
- xi does not have to be sorted; the function will sort the points by xi.
- Raises ValueError for length mismatches or insufficient points.
"""

from typing import Sequence, Union
import numpy as np


def c_spline(xi: Sequence[float], yi: Sequence[float], x_query: Union[float, Sequence[float]]):
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)

    if xi.ndim != 1 or yi.ndim != 1:
        raise ValueError("xi and yi must be 1D sequences")
    if xi.size != yi.size:
        raise ValueError("xi and yi must have the same length")
    n = xi.size
    if n < 2:
        raise ValueError("At least two data points are required for spline interpolation")

    # Sort by xi so we can assume increasing x
    order = np.argsort(xi)
    xi = xi[order]
    yi = yi[order]

    # Allocate arrays for second derivatives (y2) and temporary u
    y2 = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)  # using n for simplicity; only indices 1..n-2 used

    # Natural spline boundary conditions: y2[0] = u[0] = 0 (already zero)
    # Decomposition loop for the tridiagonal system (forward sweep)
    for i in range(1, n - 1):
        sig = (xi[i] - xi[i - 1]) / (xi[i + 1] - xi[i - 1])
        p = sig * y2[i - 1] + 2.0
        # store the decomposed factor
        y2[i] = (sig - 1.0) / p
        # compute the right-hand-side term
        d1 = (yi[i + 1] - yi[i]) / (xi[i + 1] - xi[i])
        d0 = (yi[i] - yi[i - 1]) / (xi[i] - xi[i - 1])
        u[i] = (6.0 * (d1 - d0) / (xi[i + 1] - xi[i - 1]) - sig * u[i - 1]) / p

    # Natural spline boundary on the high end: qn = un = 0 => y2[n-1] = 0
    y2[-1] = 0.0

    # Back-substitution loop for y2 (second derivatives)
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]

    # Interpolation helper for a single x value
    def interp_one(xv: float) -> float:
        # Extrapolate flat at boundaries (match the VBA which effectively clamps)
        if xv <= xi[0]:
            return float(yi[0])
        if xv >= xi[-1]:
            return float(yi[-1])

        # locate the interval using binary search
        klo = np.searchsorted(xi, xv) - 1
        khi = klo + 1

        h = xi[khi] - xi[klo]
        if h == 0:
            raise ValueError("Two identical xi values detected")

        a = (xi[khi] - xv) / h
        b = (xv - xi[klo]) / h

        result = (a * yi[klo] + b * yi[khi] +
                  ((a**3 - a) * y2[klo] + (b**3 - b) * y2[khi]) * (h**2) / 6.0)
        return float(result)

    # Handle scalar or vector x_query
    if np.isscalar(x_query):
        return interp_one(float(x_query))
    else:
        xq = np.asarray(x_query, dtype=float)
        return np.array([interp_one(val) for val in xq], dtype=float)


# Simple usage example
if __name__ == "__main__":
    # sample data

    xs = [1, 2, 3, 4]
    ys = [0.1365, 0.1325, 0.1300, 0.1250]
    print("interp at 3.5:", c_spline(xs, ys, 1.13))