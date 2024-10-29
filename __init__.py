"""
Oil Well Calculator
A comprehensive tool for oil well analysis and prediction
"""

__version__ = "1.0.0"

from src.calculations import (
    calculate_decline_curve,
    calculate_water_cut,
    calculate_operating_costs,
    calculate_npv
)

from src.models import (
    EquipmentComponent,
    EnvironmentalRegulation,
    TaxCalculator
)

from src.utils import (
    MonteCarloSimulator,
    create_production_plot,
    load_config
)