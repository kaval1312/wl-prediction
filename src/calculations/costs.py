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
    inflation_factor = (1 + inflation_rate/12) ** np.arange(months)
    
    # Calculate water production
    water_production = production * water_cut
    
    # Calculate cost components
    direct_opex = production * base_cost * inflation_factor
    water_costs = water_production * water_handling_cost * inflation_factor
    overhead = production * base_cost * 0.1 * inflation_factor
    
    total_costs = direct_opex + water_costs + overhead
    
    cost_components = {
        'direct_opex': direct_opex,
        'water_costs': water_costs,
        'overhead': overhead
    }
    
    return total_costs, cost_components

def calculate_maintenance_costs(
    equipment_age: float,
    operating_hours: float,
    equipment_health: float,
    base_cost: float = 5000.0,
    complexity_factor: float = 1.0
) -> Tuple[float, Dict]:
    """
    Calculate maintenance costs based on equipment condition.
    
    Args:
        equipment_age: Age in years
        operating_hours: Operating hours
        equipment_health: Health score (0-100)
        base_cost: Base maintenance cost
        complexity_factor: Equipment complexity multiplier
    
    Returns:
        Tuple of (total_cost, cost_breakdown)
    """
    # Age factor
    age_factor = 1 + 0.1 * equipment_age
    
    # Usage factor
    usage_factor = 1 + 0.05 * (operating_hours / 8760)  # 8760 hours per year
    
    # Health factor
    health_factor = 2 - equipment_health/100
    
    # Calculate components
    routine = base_cost * age_factor
    preventive = base_cost * 0.5 * usage_factor
    corrective = base_cost * health_factor * complexity_factor
    
    total_cost = routine + preventive + corrective
    
    cost_breakdown = {
        'routine': routine,
        'preventive': preventive,
        'corrective': corrective
    }
    
    return total_cost, cost_breakdown