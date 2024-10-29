import numpy as np
from typing import Tuple, Optional

def calculate_decline_curve(
    initial_rate: float,
    decline_rate: float,
    months: int,
    b_factor: float = 0
) -> np.ndarray:
    """
    Calculate production using hyperbolic decline curve.
    
    Args:
        initial_rate: Initial production rate (bbl/day)
        decline_rate: Monthly decline rate (fraction)
        months: Number of months to forecast
        b_factor: Hyperbolic decline factor (0=exponential)
    """
    time = np.arange(months)
    if b_factor == 0:
        # Exponential decline
        production = initial_rate * np.exp(-decline_rate * time)
    else:
        # Hyperbolic decline
        production = initial_rate / (1 + b_factor * decline_rate * time) ** (1/b_factor)
    return production

def forecast_production(
    historical_data: np.ndarray,
    forecast_months: int,
    method: str = 'decline_curve'
) -> Tuple[np.ndarray, dict]:
    """
    Forecast future production using various methods.
    
    Args:
        historical_data: Historical production data
        forecast_months: Number of months to forecast
        method: Forecasting method ('decline_curve', 'arima', etc.)
    """
    # Calculate decline rate from historical data
    decline_rate = -np.log(historical_data[-1] / historical_data[0]) / len(historical_data)
    
    # Initial rate from last historical point
    initial_rate = historical_data[-1]
    
    # Generate forecast
    forecast = calculate_decline_curve(initial_rate, decline_rate, forecast_months)
    
    metrics = {
        'decline_rate': decline_rate * 12,  # Annual decline rate
        'initial_rate': initial_rate,
        'final_rate': forecast[-1],
        'cumulative_production': np.sum(forecast)
    }
    
    return forecast, metrics

def calculate_water_cut(
    elapsed_months: int,
    initial_water_cut: float = 0.1,
    max_water_cut: float = 0.95,
    breakthrough_month: Optional[int] = None
) -> np.ndarray:
    """
    Calculate water cut progression over time.
    
    Args:
        elapsed_months: Number of months
        initial_water_cut: Initial water cut fraction
        max_water_cut: Maximum water cut fraction
        breakthrough_month: Month when water breakthrough occurs
    """
    if breakthrough_month is None:
        breakthrough_month = elapsed_months // 3
    
    time = np.arange(elapsed_months)
    water_cut = initial_water_cut + (max_water_cut - initial_water_cut) * \
                (1 / (1 + np.exp(-0.1 * (time - breakthrough_month))))
    
    return water_cut
