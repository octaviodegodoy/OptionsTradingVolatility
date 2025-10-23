"""
Simplified ARIMA model for interest rate forecasting.
Returns just the final prediction value with date.

Dependencies:
    pip install numpy pandas statsmodels
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Optional

# Import ARIMA from statsmodels
from statsmodels.tsa.arima.model import ARIMA


def predict_last_rate(
    rates: List[float],
    periods_ahead: int = 1,
    dates: Optional[List[Union[str, datetime]]] = None,
    order: Tuple[int, int, int] = (1, 1, 0)
) -> Tuple[float, Optional[datetime]]:
    """
    Predict the last (final) interest rate value using ARIMA.
    
    Parameters:
    -----------
    rates : List[float]
        Historical interest rates (most recent last)
    periods_ahead : int
        Number of periods to forecast ahead (default: 1)
    dates : Optional[List]
        Dates corresponding to historical rates (optional)
    order : Tuple[int, int, int]
        ARIMA order as (p,d,q)
        
    Returns:
    --------
    Tuple[float, Optional[datetime]]: 
        The final predicted rate and its date (if dates were provided)
    """
    # Convert input to numpy array
    rates_arr = np.array(rates)
    
    # Fit ARIMA model
    try:
        model = ARIMA(rates_arr, order=order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast_result = model_fit.forecast(steps=periods_ahead)
        
        # Get the last prediction
        last_prediction = forecast_result[-1]
        
        # If dates are provided, calculate the predicted date
        last_date = None
        if dates is not None:
            # Convert all dates to datetime if they're strings
            if isinstance(dates[0], str):
                dates = [pd.to_datetime(d) for d in dates]
            
            # Get the last historical date
            last_historical_date = dates[-1]
            
            # Infer frequency from historical dates
            if len(dates) >= 2:
                deltas = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                median_days = int(np.median(deltas))
                # Calculate the forecast date
                last_date = last_historical_date + timedelta(days=median_days * periods_ahead)
            else:
                # Default to 30 days if we can't infer
                last_date = last_historical_date + timedelta(days=30 * periods_ahead)
        
        return last_prediction, last_date
        
    except Exception as e:
        print(f"Error in ARIMA forecasting: {e}")
        return None, None


# Example usage
if __name__ == "__main__":
    # Sample data - recent interest rates
    rates = [13.507,15.165,14.349,14.368,14.249,13.275,13.504,13.452,13.57,13.205,13.255,13.33]
    
    # Current date (from user input)
    current_date = datetime(2025, 10, 18, 20, 2, 21)
    
    # Generate historical dates (monthly)
    dates = [current_date - timedelta(days=30*i) for i in range(len(rates)-1, -1, -1)]
    
    # Predict the next rate value
    prediction, pred_date = predict_last_rate(
        rates=rates,
        periods_ahead=200,  # Just twenty periods ahead
        dates=dates,
        order=(1, 1, 0)   # ARIMA(1,1,0) - simple AR model with differencing
    )
    
    # Print result
    if prediction is not None:
        print(f"Last Prediction: {prediction:.4f}% on {pred_date.strftime('%Y-%m-%d')}")
    else:
        print("Prediction failed")