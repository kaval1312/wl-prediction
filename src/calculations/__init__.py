"""
Calculations package for oil well calculator.
Contains core calculation functions for production, costs, taxes, and financial metrics.
"""

from .production import (
    calculate_decline_curve,
    calculate_water_cut,
    calculate_gas_production,
    calculate_reservoir_pressure,
    analyze_production_metrics,
    forecast_production_scenarios
)

from .costs import (
    calculate_operating_costs,
    calculate_maintenance_costs,
    calculate_workover_costs,
    calculate_total_costs
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
    calculate_profitability_metrics,
    calculate_break_even_price,
    sensitivity_analysis
)

__all__ = [
    # Production
    'calculate_decline_curve',
    'calculate_water_cut',
    'calculate_gas_production',
    'calculate_reservoir_pressure',
    'analyze_production_metrics',
    'forecast_production_scenarios',

    # Costs
    'calculate_operating_costs',
    'calculate_maintenance_costs',
    'calculate_workover_costs',
    'calculate_total_costs',

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
    'calculate_profitability_metrics',
    'calculate_break_even_price',
    'sensitivity_analysis'
]