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

    Returns:
        Array of monthly production rates
    """
    time = np.arange(months)
    if b_factor == 0:
        # Exponential decline
        production = initial_rate * np.exp(-decline_rate * time)
    else:
        # Hyperbolic decline
        production = initial_rate / (1 + b_factor * decline_rate * time) ** (1 / b_factor)
    return production


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

    Returns:
        Array of monthly water cut values
    """
    if breakthrough_month is None:
        breakthrough_month = elapsed_months // 3

    time = np.arange(elapsed_months)
    water_cut = initial_water_cut + (max_water_cut - initial_water_cut) * \
                (1 / (1 + np.exp(-0.1 * (time - breakthrough_month))))

    return water_cut


def calculate_gas_production(
        oil_rate: np.ndarray,
        gor: float = 1000,
        decline_rate: float = 0.05
) -> np.ndarray:
    """
    Calculate associated gas production.

    Args:
        oil_rate: Array of oil production rates
        gor: Gas-oil ratio (scf/bbl)
        decline_rate: Monthly GOR decline rate

    Returns:
        Array of gas production rates (mcf/d)
    """
    time = np.arange(len(oil_rate))
    declining_gor = gor * np.exp(-decline_rate * time)
    gas_rate = oil_rate * declining_gor / 1000  # Convert to mcf/d
    return gas_rate


def calculate_reservoir_pressure(
        initial_pressure: float,
        months: int,
        depletion_rate: float = 0.02
) -> np.ndarray:
    """
    Calculate reservoir pressure decline.

    Args:
        initial_pressure: Initial reservoir pressure (psi)
        months: Number of months to forecast
        depletion_rate: Monthly pressure depletion rate

    Returns:
        Array of monthly reservoir pressures
    """
    time = np.arange(months)
    pressure = initial_pressure * np.exp(-depletion_rate * time)
    return pressure


def analyze_production_metrics(
        production: np.ndarray,
        water_cut: np.ndarray,
        oil_price: float,
        opex: float
) -> dict:
    """
    Calculate key production metrics.

    Args:
        production: Array of production rates
        water_cut: Array of water cut values
        oil_price: Oil price per barrel
        opex: Operating cost per barrel

    Returns:
        Dictionary of production metrics
    """
    oil_production = production * (1 - water_cut)
    water_production = production * water_cut

    metrics = {
        'cumulative_oil': np.sum(oil_production),
        'cumulative_water': np.sum(water_production),
        'average_rate': np.mean(oil_production),
        'peak_rate': np.max(oil_production),
        'final_rate': oil_production[-1],
        'decline_rate': (oil_production[0] - oil_production[-1]) / oil_production[0] / len(production) * 12,
        'water_cut_final': water_cut[-1],
        'gross_revenue': np.sum(oil_production * oil_price),
        'total_opex': np.sum(production * opex),
        'unit_cost': np.sum(production * opex) / np.sum(oil_production)
    }

    return metrics


def forecast_production_scenarios(
        base_production: np.ndarray,
        scenarios: dict = None
) -> dict:
    """
    Generate production scenarios.

    Args:
        base_production: Base case production forecast
        scenarios: Dictionary of scenario adjustments

    Returns:
        Dictionary of production scenarios
    """
    if scenarios is None:
        scenarios = {
            'low': 0.8,
            'high': 1.2
        }

    forecasts = {
        'base': base_production
    }

    for name, factor in scenarios.items():
        forecasts[name] = base_production * factor

    return forecasts