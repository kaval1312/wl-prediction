"""
Calculations package for oil well calculator.
Contains core calculation functions for production, costs, taxes, and financial metrics.
"""

from .production import (
    calculate_decline_curve,
    calculate_water_cut,  # Changed from calculate_water_cut_increase
    forecast_production   # Added this function
)

from .costs import (
    calculate_operating_costs,
    calculate_maintenance_costs
)

from .taxes import (
    calculate_tax_obligations,
    calculate_depletion_allowance,
    calculate_severance_tax,
    calculate_carbon_tax
)

from .npv import (
    calculate_npv,
    calculate_irr,
    calculate_payback_period,
    calculate_unit_costs,
    calculate_profitability_index,
    calculate_break_even_price,
    sensitivity_analysis
)

__all__ = [
    # Production
    'calculate_decline_curve',
    'calculate_water_cut',
    'forecast_production',
    
    # Costs
    'calculate_operating_costs',
    'calculate_maintenance_costs',
    
    # Taxes
    'calculate_tax_obligations',
    'calculate_depletion_allowance',
    'calculate_severance_tax',
    'calculate_carbon_tax',
    
    # Financial
    'calculate_npv',
    'calculate_irr',
    'calculate_payback_period',
    'calculate_unit_costs',
    'calculate_profitability_index',
    'calculate_break_even_price',
    'sensitivity_analysis'
]