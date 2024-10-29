import numpy as np
from typing import Tuple, Dict


def calculate_operating_costs(
        production: np.ndarray,
        water_cut: np.ndarray,
        base_cost: float = 10.0,
        water_handling_cost: float = 2.0,
        inflation_rate: float = 0.03
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate operating costs for production.

    Args:
        production: Daily oil production rates
        water_cut: Water cut fraction
        base_cost: Base operating cost per barrel
        water_handling_cost: Cost per barrel of water
        inflation_rate: Annual inflation rate

    Returns:
        Tuple of (total_costs, cost_components)
    """
    months = len(production)
    inflation_factor = (1 + inflation_rate / 12) ** np.arange(months)

    # Calculate water production
    water_production = production * water_cut

    # Calculate cost components
    direct_opex = production * base_cost * inflation_factor
    water_costs = water_production * water_handling_cost * inflation_factor
    overhead = production * base_cost * 0.1 * inflation_factor
    maintenance = production * base_cost * 0.15 * inflation_factor

    total_costs = direct_opex + water_costs + overhead + maintenance

    cost_components = {
        'direct_opex': direct_opex,
        'water_costs': water_costs,
        'overhead': overhead,
        'maintenance': maintenance
    }

    return total_costs, cost_components


def calculate_maintenance_costs(
        months: int,
        base_cost: float = 5000.0,
        inflation_rate: float = 0.03,
        equipment_age: float = 0,
        well_depth: float = 5000,
        schedule_frequency: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate maintenance costs and schedule.

    Args:
        months: Number of months to calculate
        base_cost: Base maintenance cost
        inflation_rate: Annual inflation rate
        equipment_age: Age of equipment in years
        well_depth: Well depth in feet
        schedule_frequency: Months between maintenance

    Returns:
        Tuple of (maintenance_costs, maintenance_events)
    """
    # Initialize arrays
    maintenance_costs = np.zeros(months)
    maintenance_events = np.zeros(months)

    # Age factor increases costs as equipment ages
    age_factor = 1 + 0.1 * equipment_age

    # Depth factor increases costs for deeper wells
    depth_factor = 1 + 0.1 * (well_depth / 5000 - 1)

    # Calculate scheduled maintenance costs
    schedule_months = np.arange(0, months, schedule_frequency)
    maintenance_events[schedule_months] = 1

    # Base maintenance cost with factors
    adjusted_cost = base_cost * age_factor * depth_factor

    # Apply maintenance costs
    maintenance_costs[schedule_months] = adjusted_cost

    # Apply inflation
    inflation_factor = (1 + inflation_rate / 12) ** np.arange(months)
    maintenance_costs *= inflation_factor

    return maintenance_costs, maintenance_events


def calculate_workover_costs(
        production: np.ndarray,
        failure_probability: np.ndarray,
        base_cost: float = 50000.0,
        depth_factor: float = 1.0
) -> Tuple[np.ndarray, int]:
    """
    Calculate workover costs based on equipment failures.

    Args:
        production: Daily production rates
        failure_probability: Equipment failure probabilities
        base_cost: Base workover cost
        depth_factor: Cost multiplier for well depth

    Returns:
        Tuple of (workover_costs, number_of_workovers)
    """
    # Generate random failures based on probability
    failures = np.random.binomial(1, failure_probability)

    # Calculate costs for each failure
    workover_costs = failures * base_cost * depth_factor

    return workover_costs, np.sum(failures)


def calculate_total_costs(
        production: np.ndarray,
        water_cut: np.ndarray,
        opex_params: dict,
        maintenance_params: dict
) -> Dict[str, np.ndarray]:
    """
    Calculate all cost components.

    Args:
        production: Daily production rates
        water_cut: Water cut values
        opex_params: Operating cost parameters
        maintenance_params: Maintenance cost parameters

    Returns:
        Dictionary of cost arrays
    """
    # Calculate operating costs
    operating_costs, opex_components = calculate_operating_costs(
        production=production,
        water_cut=water_cut,
        **opex_params
    )

    # Calculate maintenance costs
    maintenance_costs, maintenance_events = calculate_maintenance_costs(
        months=len(production),
        **maintenance_params
    )

    # Combine all costs
    total_costs = operating_costs + maintenance_costs

    return {
        'total': total_costs,
        'operating': operating_costs,
        'maintenance': maintenance_costs,
        'maintenance_events': maintenance_events,
        **opex_components
    }