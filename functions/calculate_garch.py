import logging
from unittest import result
from scipy import stats
import numpy as np
from arch import arch_model
import pandas as pd
from mt5_connector import MT5Connector


class GARCHCalculation:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mt5_conn = MT5Connector()
        self.logger.info("GARCH Calculation instance created.")
        self.data = None
        self.returns = None
        self.model = None   
        self.fitted_model = None

    def _garch_negloglik_from_raw(theta_raw: np.ndarray, rets: np.ndarray, dist: str = "t") -> float:
        params = _raw_to_model(theta_raw, dist=dist)
        mu = params["mu"]
        omega = params["omega"]
        alpha = params["alpha"]
        beta = params["beta"]
        eps = rets - mu
        n = len(rets)

        # initialize sigma2[0]
        sample_var = np.var(eps, ddof=1) if n > 1 else 1e-6
        denom = 1.0 - alpha - beta
        if denom > 1e-8:
            sigma2_0 = max(omega / denom, _VAR_FLOOR)
        else:
            sigma2_0 = max(sample_var, _VAR_FLOOR)

        sigma2 = np.empty(n, dtype=float)
        sigma2[0] = sigma2_0
        for t in range(1, n):
            sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
            if sigma2[t] < _VAR_FLOOR:
                sigma2[t] = _VAR_FLOOR

        if dist in ("normal", "gaussian"):
            ll = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + (eps ** 2) / sigma2)
            return -np.sum(ll)
        elif dist == "t":
            nu = params.get("nu", 8.0)
            if nu <= 2:
                return 1e12
            const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi)
            z = (eps ** 2) / (sigma2 * (nu - 2.0))
            ll_t = const - 0.5 * np.log(sigma2) - ((nu + 1.0) / 2.0) * np.log1p(z)
            return -np.sum(ll_t)
        else:
            raise ValueError("dist must be 't' or 'normal'")

    def fit_garch_mle_simple(returns_or_prices: Union[pd.Series, pd.DataFrame, np.ndarray], dist: str = "t",
                         options: Optional[dict] = None) -> Dict[str, Any]:
        """
        Fit GARCH(1,1) by MLE using transform params. Returns fitted model params and optimizer info.
        """
        if options is None:
            options = {"maxiter": 5000, "disp": False}

        rets = _to_returns(returns_or_prices)
        n = len(rets)
        if n < 30:
            raise ValueError("Need at least ~30 observations to fit GARCH(1,1)")

        mu0 = float(np.mean(rets))
        alpha0 = 0.05
        beta0 = 0.9
        # adjust if alpha0+beta0 >=1
        if alpha0 + beta0 >= 0.999:
            alpha0 = 0.05
            beta0 = 0.85
        sample_var = np.var(rets, ddof=1)
        omega0 = max(1e-8, sample_var * (1.0 - alpha0 - beta0))
        denom0 = 1.0 - alpha0 - beta0
        exp_a0 = alpha0 / denom0
        exp_b0 = beta0 / denom0
        a0 = np.log(max(exp_a0, 1e-8))
        b0 = np.log(max(exp_b0, 1e-8))
        c0 = np.log(max(omega0, 1e-12))
        if dist == "t":
            d0 = np.log(4.0)  # nu = 2 + exp(d0) ~ 6
            theta0 = np.array([mu0, a0, b0, c0, d0], dtype=float)
        else:
            theta0 = np.array([mu0, a0, b0, c0], dtype=float)

        obj = lambda th: _garch_negloglik_from_raw(th, rets, dist=dist)
        opt = minimize(obj, theta0, method="L-BFGS-B", options=options)
        theta_hat = opt.x
        params = _raw_to_model(theta_hat, dist=dist)
        result = {
            "params": params,
            "raw_params": theta_hat,
            "opt_result": opt,
            "nobs": n,
            "rets": rets
        }
        return result


    def predict_vol_from_prices_mle(prices: Union[pd.Series, pd.DataFrame, np.ndarray],
                                days_to_expiry: int = 1,
                                dist: str = "t",
                                days_per_year: int = 252,
                                options: Optional[dict] = None) -> Dict[str, Any]:
        """
        Fit GARCH(1,1) by MLE and return predicted volatility for the given days_to_expiry.

        Returns:
        {
            "last_cond_vol_daily": float (decimal),
            "period_vol_T": float (decimal),   # Std over T days
            "annualized_vol_over_T": float (decimal),
            "model_params": dict,
            "daily_variance_forecasts": np.ndarray (length T)
        }
        """
        fit = fit_garch_mle_simple(prices, dist=dist, options=options)
        params = fit["params"]
        rets = fit["rets"]
        # conditional variances over sample
        sigma2_series = _compute_conditional_variances_from_params(rets, params)
        last_sigma2 = float(sigma2_series[-1])
        last_eps_sq = float((rets - params["mu"])[-1] ** 2)

        # Forecast T next daily variances
        T = int(days_to_expiry)
        var_forecasts = forecast_next_T_variances(last_sigma2, last_eps_sq, params, T)
        # ensure positive
        var_forecasts = np.maximum(var_forecasts, 0.0)

        # T-day variance = sum of daily variances (approx for aggregated returns)
        var_T = float(np.sum(var_forecasts))
        std_T = float(np.sqrt(var_T))
        # annualize: convert T-day std to annualized vol
        annualized_vol = std_T * np.sqrt(days_per_year / float(T))

        out = {
            "last_cond_vol_daily": float(np.sqrt(last_sigma2)),  # daily std (decimal)
            "period_vol_T": std_T,
            "annualized_vol_over_T": annualized_vol,
            "model_params": params,
            "daily_variance_forecasts": var_forecasts,
            "fit_info": {"nobs": fit["nobs"], "optimizer": fit["opt_result"]}
        }
    return out