"""
ARIMA model for interest rate forecasting.

This module provides a clean, focused implementation of ARIMA models
for interest rate forecasting with clear visualization and diagnostics.

Dependencies:
    pip install numpy pandas matplotlib statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Dict, Optional

# Import ARIMA from statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ARIMAForecaster:
    """ARIMA model for interest rate forecasting."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 0)
    ):
        """
        Initialize the ARIMA forecaster.

        Parameters:
        -----------
        order : Tuple[int, int, int]
            ARIMA order as (p, d, q)
            - p: AR (autoregressive) order
            - d: I (integration/differencing) order
            - q: MA (moving average) or der
        """
        self.order = order
        self.model = None
        self.model_fit = None
        self.data = None
        self.date_index = None
        self.is_fitted = False

    def fit(
        self,
        rates: Union[List[float], np.ndarray, pd.Series],
        dates: Optional[List[Union[str, datetime]]] = None
    ) -> "ARIMAForecaster":
        """
        Fit the ARIMA model to historical interest rate data.

        Parameters:
        -----------
        rates : array-like
            Historical interest rates
        dates : list of datetime, optional
            Dates corresponding to each rate value

        Returns:
        --------
        self : returns an instance of self
        """
        # Convert rates to numpy array if needed
        if isinstance(rates, pd.Series):
            if dates is None and isinstance(rates.index, pd.DatetimeIndex):
                # Use Series index as dates
                self.date_index = rates.index
                self.data = rates.values
            else:
                self.data = rates.values
                if dates is not None:
                    self.date_index = pd.DatetimeIndex(dates)
                else:
                    self.date_index = None
        else:
            self.data = np.array(rates)
            if dates is not None:
                self.date_index = pd.DatetimeIndex(dates)
            else:
                self.date_index = None

        # Fit ARIMA model
        try:
            self.model = ARIMA(self.data, order=self.order)
            self.model_fit = self.model.fit()
            self.is_fitted = True
            return self
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            self.is_fitted = False
            return self

    def forecast(
        self,
        periods: int,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> Dict:
        """
        Generate interest rate forecasts.

        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals (default 0.05 for 95% CI)

        Returns:
        --------
        Dict containing forecast results:
            - 'forecast': array of point forecasts
            - 'lower_ci': lower confidence bound (if return_conf_int=True)
            - 'upper_ci': upper confidence bound (if return_conf_int=True)
            - 'forecast_dates': dates for forecasts (if original data had dates)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        # Get forecast
        forecast_result = self.model_fit.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean

        result = {'forecast': forecast_mean}

        # Add confidence intervals if requested
        if return_conf_int:
            conf_int = forecast_result.conf_int(alpha=alpha)
            # Access confidence intervals as NumPy array
            result['lower_ci'] = conf_int[:, 0]
            result['upper_ci'] = conf_int[:, 1]

        # Generate forecast dates if we have a date index
        if self.date_index is not None:
            last_date = self.date_index[-1]

            # Infer frequency from the data
            if len(self.date_index) >= 2:
                # Calculate most common frequency
                deltas = np.diff(self.date_index.astype(int)) / 1e9 / 86400  # convert to days
                median_days = np.median(deltas)

                if median_days < 1.5:
                    # Daily data
                    forecast_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=periods,
                        freq='D'
                    )
                elif 25 <= median_days <= 32:
                    # Monthly data
                    forecast_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=periods,
                        freq='MS'  # Month start
                    )
                elif 85 <= median_days <= 95:
                    # Quarterly data
                    forecast_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=periods,
                        freq='QS'  # Quarter start
                    )
                else:
                    # Default to the exact median days
                    forecast_dates = [last_date + timedelta(days=int(median_days*i))
                                     for i in range(1, periods+1)]
                    forecast_dates = pd.DatetimeIndex(forecast_dates)
            else:
                # Default to daily if we can't infer
                forecast_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=periods,
                    freq='D'
                )

            result['forecast_dates'] = forecast_dates

        return result

    def plot_forecast(
        self,
        forecast_result: Dict,
        title: str = "ARIMA Interest Rate Forecast",
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot historical data with forecasted values and confidence intervals.

        Parameters:
        -----------
        forecast_result : Dict
            Output from forecast() method
        title : str
            Plot title
        figsize : Tuple[int, int]
            Figure size

        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Determine x-axis values
        if self.date_index is not None and 'forecast_dates' in forecast_result:
            # Date-based x-axis
            x_hist = self.date_index
            # Convert forecast dates to DatetimeIndex if they are not already
            x_forecast = pd.DatetimeIndex(forecast_result['forecast_dates'])
            use_dates = True
        else:
            # Index-based x-axis
            x_hist = np.arange(len(self.data))
            x_forecast = np.arange(len(self.data), len(self.data) + len(forecast_result['forecast']))
            use_dates = False

        # Plot historical data
        ax.plot(x_hist, self.data, 'b.-', markersize=8, linewidth=2, label='Historical')

        # Plot forecast
        ax.plot(x_forecast, forecast_result['forecast'], 'r.-', markersize=8, linewidth=2, label='Forecast')

        # Plot confidence intervals if available
        if 'lower_ci' in forecast_result and 'upper_ci' in forecast_result:
            ax.fill_between(
                x_forecast,
                forecast_result['lower_ci'],
                forecast_result['upper_ci'],
                color='r',
                alpha=0.2,
                label='95% Confidence Interval'
            )

        # Set up axis labels and title
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Interest Rate', fontsize=12)

        if use_dates:
            ax.set_xlabel('Date', fontsize=12)
            fig.autofmt_xdate()  # Rotate date labels

            # Improve date formatting
            if len(x_hist) + len(x_forecast) > 20:
                # Many dates - use a locator to avoid overcrowding
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            ax.set_xlabel('Period', fontsize=12)

        # Add vertical line at the forecast start
        ax.axvline(
            x=x_hist[-1] if use_dates else len(self.data) - 0.5,
            color='gray',
            linestyle='--',
            alpha=0.7
        )

        # Add ARIMA order to the plot
        p, d, q = self.order
        order_text = f"ARIMA({p},{d},{q})"
        plt.figtext(
            0.02, 0.02, order_text,
            fontsize=10, ha='left'
        )

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot model diagnostics including:
        - Residuals over time
        - Residual histogram
        - Q-Q plot
        - Autocorrelation function of residuals

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size

        Returns:
        --------
        matplotlib.figure.Figure
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        fig = plt.figure(figsize=figsize)

        # Residuals over time
        ax1 = plt.subplot(221)
        residuals = self.model_fit.resid
        if self.date_index is not None:
            ax1.plot(self.date_index, residuals, 'o-')
            fig.autofmt_xdate()
        else:
            ax1.plot(residuals, 'o-')
        ax1.set_title('Residuals')
        ax1.axhline(y=0, color='r', linestyle='-')

        # Histogram of residuals
        ax2 = plt.subplot(222)
        from scipy import stats
        ax2.hist(residuals, bins=20, density=True, alpha=0.7)
        ax2.set_title('Histogram of Residuals')

        # Q-Q plot
        ax3 = plt.subplot(223)
        stats.probplot(residuals, plot=ax3)

        # Autocorrelation of residuals
        ax4 = plt.subplot(224)
        plot_acf(residuals, ax=ax4, lags=min(20, len(residuals)//2))
        ax4.set_title('Autocorrelation of Residuals')

        plt.tight_layout()
        return fig

    def suggest_order(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        plot: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ) -> Tuple[int, int, int]:
        """
        Suggest ARIMA order based on ACF, PACF and stationarity tests.

        Parameters:
        -----------
        max_p : int
            Maximum AR order to consider
        max_d : int
            Maximum differencing order to consider
        max_q : int
            Maximum MA order to consider
        plot : bool
            Whether to plot ACF and PACF
        figsize : Tuple[int, int]
            Figure size for plots

        Returns:
        --------
        Tuple[int, int, int]: Suggested ARIMA order (p, d, q)
        """
        if self.data is None:
            raise ValueError("No data available. Call fit() first.")

        # Determine d (differencing) using ADF test
        d = 0
        data = self.data.copy()

        # Perform ADF test on original data
        adf_result = adfuller(data)
        p_value = adf_result[1]

        # If non-stationary, try differencing until stationary
        while p_value > 0.05 and d < max_d:
            d += 1
            diff_data = np.diff(data, n=1)
            adf_result = adfuller(diff_data)
            p_value = adf_result[1]
            data = diff_data

        # Calculate ACF and PACF
        if d > 0:
            diff_data = np.diff(self.data, n=d)
        else:
            diff_data = self.data.copy()

        acf_values = acf(diff_data, nlags=max_p+max_q, fft=True)
        pacf_values = pacf(diff_data, nlags=max_p, method='ols')

        # Determine q from ACF
        q = 0
        acf_cutoff = 1.96 / np.sqrt(len(diff_data))  # 95% confidence interval

        for i in range(1, min(max_q+1, len(acf_values))):
            if abs(acf_values[i]) > acf_cutoff:
                q = i

        # Determine p from PACF
        p = 0
        pacf_cutoff = 1.96 / np.sqrt(len(diff_data))  # 95% confidence interval

        for i in range(1, min(max_p+1, len(pacf_values))):
            if abs(pacf_values[i]) > pacf_cutoff:
                p = i

        # Default to AR(1) if no significant lags found
        if p == 0 and q == 0:
            p = 1

        # Plot ACF and PACF if requested
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            plot_acf(diff_data, ax=ax1, lags=max_p+max_q)
            ax1.axhline(y=acf_cutoff, color='r', linestyle='--', alpha=0.5)
            ax1.axhline(y=-acf_cutoff, color='r', linestyle='--', alpha=0.5)
            ax1.set_title(f'Autocorrelation Function (Differencing Order = {d})')

            plot_pacf(diff_data, ax=ax2, lags=max_p, method='ols')
            ax2.axhline(y=pacf_cutoff, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=-pacf_cutoff, color='r', linestyle='--', alpha=0.5)
            ax2.set_title('Partial Autocorrelation Function')

            plt.tight_layout()
            plt.show()

        suggested_order = (p, d, q)
        print(f"Suggested ARIMA order: {suggested_order}")
        print(f"ADF Test p-value after {d} differencing: {p_value:.4f}")

        return suggested_order


def forecast_from_recent_rates(
    rates: List[float],
    forecast_periods: int = 2,
    dates: Optional[List[Union[str, datetime]]] = None,
    order: Optional[Tuple[int, int, int]] = None,
    auto_order: bool = True,
    confidence_level: float = 0.95,
    plot: bool = True,
) -> Dict:
    """
    Convenience function to forecast interest rates from recent data.

    Parameters:
    -----------
    rates : List[float]
        Recent interest rates (most recent last)
    forecast_periods : int
        Number of periods to forecast
    dates : Optional[List]
        Dates corresponding to rates
    order : Tuple[int, int, int]
        ARIMA order as (p, d, q), if None and auto_order=True will be determined automatically
    auto_order : bool
        Whether to automatically determine ARIMA order
    confidence_level : float
        Confidence level for forecast intervals (e.g., 0.95 for 95%)
    plot : bool
        Whether to plot the forecast

    Returns:
    --------
    Dict with forecast results
    """
    # Create forecaster
    forecaster = ARIMAForecaster(order=(1, 1, 0))  # Default order

    # Fit to data
    forecaster.fit(rates, dates)

    # Determine optimal order if requested
    if auto_order:
        suggested_order = forecaster.suggest_order(plot=False)

        # Update model with suggested order if different
        if suggested_order != forecaster.order:
            forecaster = ARIMAForecaster(order=suggested_order)
            forecaster.fit(rates, dates)

    # Use provided order if specified
    elif order is not None:
        forecaster = ARIMAForecaster(order=order)
        forecaster.fit(rates, dates)

    # Generate forecast
    forecast_result = forecaster.forecast(
        periods=forecast_periods,
        return_conf_int=True,
        alpha=1.0 - confidence_level
    )

    # Plot if requested
    if plot:
        forecaster.plot_forecast(forecast_result)
        plt.show()

    return {
        'forecaster': forecaster,
        'forecast': forecast_result,
        'order': forecaster.order
    }


# Example usage
if __name__ == "__main__":
    # Current date from user input
    current_date = datetime(2025, 10, 18, 14, 32, 22)

    # Create historical dates (monthly data)
    n_history = 12
    historical_dates = [current_date - timedelta(days=30*i) for i in range(n_history, 0, -1)]

    # Sample interest rate data (simulated Federal Funds Rate)
    # Starting from 12 months ago
    historical_rates = [13.507,15.165,14.349,14.368,14.249,13.275,13.504,13.452,13.57,13.205,13.255,13.33]

    print("===== Interest Rate Forecasting with ARIMA =====")
    print(f"Current Date: {current_date}")
    print(f"Historical Rates: {historical_rates}")

    # Forecast using the convenience function
    print("\n1. Quick Forecast with Auto Order Selection:")
    forecast_result = forecast_from_recent_rates(
        rates=historical_rates,
        dates=historical_dates,
        forecast_periods=3,
        auto_order=True,
        plot=True
    )

    # Print forecast results
    print("\nForecasted Interest Rates:")
    if 'forecast_dates' in forecast_result['forecast']:
        for date, rate in zip(
            forecast_result['forecast']['forecast_dates'],
            forecast_result['forecast']['forecast']
        ):
            print(f"{date.strftime('%Y-%m-%d')}: {rate:.3f}%")
    else:
        for i, rate in enumerate(forecast_result['forecast']['forecast']):
            print(f"Period {i+1}: {rate:.3f}%")

    # Advanced usage with manual model building
    print("\n2. Manual Model Building and Diagnostics:")

    # Create forecaster with specific order
    forecaster = ARIMAForecaster(order=(2, 1, 1))

    # Fit to data
    forecaster.fit(historical_rates, historical_dates)

    # Plot diagnostics
    forecaster.plot_diagnostics()
    plt.tight_layout()
    plt.show()

    # Generate and plot forecast
    forecast = forecaster.forecast(periods=6)
    forecaster.plot_forecast(forecast, title="Federal Funds Rate Forecast (ARIMA)")
    plt.show()