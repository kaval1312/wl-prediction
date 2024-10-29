"""
Oil Well Calculator
A comprehensive tool for oil well analysis and prediction
Copyright (c) 2024
"""

# Version information
__version__ = "1.0.0"
__author__ = "WL Prediction"
__email__ = "support@wlprediction.com"

# Import core functionality
from src.utils.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults
)

from src.utils.economic_parameters import EconomicParameters
from src.utils.economics import EconomicAnalyzer, EconomicResults
from src.utils.equipment import EquipmentAnalyzer
from src.utils.lease_analysis import LeaseAnalyzer
from src.utils.lease_terms import LeaseTerms, AbandonmentCosts

# Import plotting functions with aliases for backward compatibility
from src.utils.plotting import (
    create_monte_carlo_analysis as create_monte_carlo_plots,
    create_production_profile as create_production_plot,
    create_cost_analysis as create_costs_plot,
    create_cashflow_analysis as create_cash_flow_plot,
    create_risk_analysis as create_risk_metrics_plot,
    create_sensitivity_analysis as create_tornado_plot,
    create_correlation_matrix as create_scatter_matrix,
    create_probability_analysis,
    create_abandonment_analysis as create_abandonment_plots,
    create_technical_analysis as create_technical_plots,
    create_equipment_analysis as create_equipment_health_plot
)

# Import calculation utilities
from src.calculations.production import (
    calculate_decline_curve,
    calculate_water_cut,
    calculate_gas_production,
    calculate_reservoir_pressure
)

from src.calculations.costs import (
    calculate_operating_costs,
    calculate_maintenance_costs,
    calculate_workover_costs
)

from src.calculations.taxes import (
    calculate_tax_obligations,
    calculate_depletion_allowance,
    calculate_severance_tax
)

from src.calculations.npv import (
    calculate_npv,
    calculate_irr,
    calculate_payback_period,
    calculate_break_even_price,
    calculate_profitability_metrics
)

# Import model classes
from src.models.equipment import EquipmentComponent, EquipmentFailure
from src.models.environmental import EnvironmentalRegulation, ComplianceViolation
from src.models.financial import TaxCalculator, FinancialMetrics

# Define all importable names
__all__ = [
    # Core classes
    'MonteCarloSimulator',
    'MonteCarloResults',
    'EconomicParameters',
    'EconomicAnalyzer',
    'EconomicResults',
    'LeaseAnalyzer',
    'LeaseTerms',
    'AbandonmentCosts',
    'EquipmentAnalyzer',

    # Plotting functions
    'create_monte_carlo_plots',
    'create_production_plot',
    'create_costs_plot',
    'create_cash_flow_plot',
    'create_risk_metrics_plot',
    'create_tornado_plot',
    'create_scatter_matrix',
    'create_probability_analysis',
    'create_abandonment_plots',
    'create_technical_plots',
    'create_equipment_health_plot',

    # Calculation functions
    'calculate_decline_curve',
    'calculate_water_cut',
    'calculate_gas_production',
    'calculate_reservoir_pressure',
    'calculate_operating_costs',
    'calculate_maintenance_costs',
    'calculate_workover_costs',
    'calculate_tax_obligations',
    'calculate_depletion_allowance',
    'calculate_severance_tax',
    'calculate_npv',
    'calculate_irr',
    'calculate_payback_period',
    'calculate_break_even_price',
    'calculate_profitability_metrics',

    # Model classes
    'EquipmentComponent',
    'EquipmentFailure',
    'EnvironmentalRegulation',
    'ComplianceViolation',
    'TaxCalculator',
    'FinancialMetrics'
]

# Configure logging
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Set up configuration
def load_configuration(config_path: str = None) -> dict:
    """Load application configuration."""
    from pathlib import Path
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent / 'src' / 'config' / 'settings.yaml'

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        return {}


# Initialize default configuration
default_config = load_configuration()


def get_version():
    """Return the current version of the package."""
    return __version__


def get_config():
    """Return the current configuration."""
    return default_config.copy()