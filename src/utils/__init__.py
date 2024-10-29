from .monte_carlo import MonteCarloSimulator, MonteCarloResults
from .economic_parameters import EconomicParameters
from .economics import EconomicAnalyzer, EconomicResults
from .equipment import EquipmentAnalyzer
from .lease_analysis import LeaseAnalyzer
from .lease_terms import LeaseTerms, AbandonmentCosts

# Import and create aliases for plotting functions
from .plotting import (
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

__all__ = [
    # Monte Carlo classes
    'MonteCarloSimulator',
    'MonteCarloResults',
    
    # Economic classes
    'EconomicParameters',
    'EconomicAnalyzer',
    'EconomicResults',
    
    # Lease classes
    'LeaseAnalyzer',
    'LeaseTerms',
    'AbandonmentCosts',
    
    # Equipment classes
    'EquipmentAnalyzer',
    
    # Plotting functions (with backward compatibility aliases)
    'create_monte_carlo_plots',      # Alias for create_monte_carlo_analysis
    'create_production_plot',        # Alias for create_production_profile
    'create_costs_plot',            # Alias for create_cost_analysis
    'create_cash_flow_plot',        # Alias for create_cashflow_analysis
    'create_risk_metrics_plot',     # Alias for create_risk_analysis
    'create_tornado_plot',          # Alias for create_sensitivity_analysis
    'create_scatter_matrix',        # Alias for create_correlation_matrix
    'create_probability_analysis',
    'create_abandonment_plots',     # Alias for create_abandonment_analysis
    'create_technical_plots',       # Alias for create_technical_analysis
    'create_equipment_health_plot'  # Alias for create_equipment_analysis
]

__version__ = '1.0.0'

def load_config():
    """Load application configuration."""
    return None
