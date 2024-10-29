"""
Oil Well Abandonment Calculator
"""

__version__ = "1.0.0"
__author__ = "Alex"


from .models import EquipmentComponent, EnvironmentalRegulation
from .calculations import (
    calculate_decline_curve,
    calculate_water_cut_increase,
    calculate_maintenance_costs,
    calculate_npv
)