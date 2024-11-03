# streamlit_app.py

import logging
from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from src.utils import (
    MonteCarloSimulator,
    EconomicParameters,
    EconomicAnalyzer,
    EquipmentAnalyzer,
    LeaseAnalyzer,
    LeaseTerms,
    AbandonmentCosts,
    create_monte_carlo_plots,
    create_production_plot,
    create_costs_plot,
    create_cash_flow_plot,
    create_risk_metrics_plot,
    create_tornado_plot,
    create_technical_plots
)


# State Management Functions
# def generate_unique_key(prefix: str) -> str:
#     """Generate a unique key for Streamlit widgets."""
#     import time
#     import random
#     return f"{prefix}_{int(time.time() * 1000)}_{random.randint(0, 1000000)}"


# def get_unique_key(prefix: str) -> str:
#     """Get a unique key and store it in session state."""
#     if 'used_keys' not in st.session_state:
#         st.session_state.used_keys = set()
#     key = generate_unique_key(prefix)
#     st.session_state.used_keys.add(key)
#     return key


def remove_all_widget_keys():
    """Remove all widget keys from session state."""
    parameter_keys = [
        'well_parameters',
        'production_parameters',
        'economic_parameters',
        'monte_carlo_parameters',
        'lease_parameters',
        'abandonment_parameters',
        'technical_parameters',
        'regulatory_parameters',
        'production_data',
        'economic_results',
        'monte_carlo_results',
        'technical_results',
        'abandonment_results',
        'current_page'
    ]

    keys_to_remove = [key for key in list(st.session_state.keys())
                      if not key.startswith('_') and key in parameter_keys]
    for key in keys_to_remove:
        del st.session_state[key]
    return keys_to_remove
def clear_session_state():
    """Clear the entire session state and reinitialize defaults."""
    st.session_state.clear()
    initialize_session_state()  # Reinitialize with defaults
def clear_all_widgets():
    """Clear all widgets and cached keys."""
    st.session_state.clear()
    initialize_session_state()  # Reinitialize with defaults
def clear_cache_and_rerun():
    """Clear cache and force rerun."""
    st.cache_data.clear()
    st.cache_resource.clear()
    clear_all_widgets()
    st.experimental_rerun()
def create_initial_data():
    """Create initial dataset structure."""
    return {
        'well_parameters': {
            'depth': 5000.0,
            'age': 5.0,
            'type': 'Oil',
            'location': 'Default Field',
            'api_number': '12345678',
            'completion_date': '2024-01-01'
        },
        'production_parameters': {
            'initial_rate': 1000.0,
            'decline_rate': 15.0,
            'forecast_months': 120,
            'water_cut': 20.0,
            'gas_oil_ratio': 1000.0
        },
        'economic_parameters': {
            'oil_price': 70.0,
            'gas_price': 3.50,
            'opex': 20.0,
            'initial_investment': 1000000.0,
            'discount_rate': 10.0,
            'working_interest': 75.0,
            'net_revenue_interest': 65.0
        },
        'technical_parameters': {
            'reservoir_pressure': 3000.0,
            'temperature': 180.0,
            'api_gravity': 35.0,
            'porosity': 20.0,
            'water_saturation': 30.0,
            'formation_volume_factor': 1.2,
            'gas_oil_ratio': 1000.0
        },
        'equipment_parameters': {
            'pump_vibration': 0.3,
            'motor_temperature': 165,
            'separator_pressure': 250,
            'last_maintenance': datetime.now() - timedelta(days=45)
        }
    }
def load_config(config_path: str = 'src/config/settings.yaml') -> dict:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            # Initialize session state with config values if needed
            if 'parameters' in config and 'defaults' in config['parameters']:
                defaults = config['parameters']['defaults']
                for key, value in defaults.items():
                    param_key = f"{key}_parameters"
                    if param_key not in st.session_state:
                        st.session_state[param_key] = value
            return config
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}
def initialize_session_state():
    """Initialize all required session state variables."""
    # Initialize current page if not present
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Production Analysis"

    # Initialize analyzers
    if 'monte_carlo' not in st.session_state:
        st.session_state.monte_carlo = MonteCarloSimulator()
    if 'economic' not in st.session_state:
        st.session_state.economic = EconomicAnalyzer()
    if 'equipment' not in st.session_state:
        st.session_state.equipment = EquipmentAnalyzer()
    if 'lease' not in st.session_state:
        st.session_state.lease = LeaseAnalyzer()

    # Initialize all parameter sets
    if 'well_parameters' not in st.session_state:
        st.session_state.well_parameters = {
            'depth': 5000.0,
            'age': 5.0,
            'type': 'Oil',
            'location': 'Default Field',
            'completion_date': '2024-01-01',
            'api_number': '12345678',
            'field_name': 'Test Field'
        }

    if 'production_parameters' not in st.session_state:
        st.session_state.production_parameters = {
            'initial_rate': 1000.0,
            'decline_rate': 15.0,
            'forecast_months': 120,
            'water_cut': 20.0,
            'gas_oil_ratio': 1000.0,
            'bottom_hole_pressure': 2000.0,
            'tubing_pressure': 500.0,
            'choke_size': 64
        }

    if 'economic_parameters' not in st.session_state:
        st.session_state.economic_parameters = {
            'oil_price': 70.0,
            'gas_price': 3.50,
            'opex': 20.0,
            'initial_investment': 1000000.0,
            'discount_rate': 10.0,
            'working_interest': 75.0,
            'net_revenue_interest': 65.0,
            'tax_rate': 21.0,
            'inflation_rate': 3.0,
            'lease_terms': 5.0,
            'abandonment_costs': 100000.0
        }

    if 'lease_parameters' not in st.session_state:
        st.session_state.lease_parameters = {
            'working_interest': 75.0,
            'net_revenue_interest': 65.0,
            'royalty_rate': 20.0,
            'lease_bonus': 50000.0,
            'lease_term_years': 5.0,
            'extension_cost': 25000.0,
            'minimum_royalty': 1000.0,
            'delay_rentals': 5000.0,
            'primary_term_end': '2029-01-01',
            'lease_expiration': '2029-01-01'
        }

    if 'abandonment_parameters' not in st.session_state:
        st.session_state.abandonment_parameters = {
            'plugging_cost_per_foot': 25.0,
            'site_restoration_cost': 50000.0,
            'equipment_removal_cost': 75000.0,
            'environmental_cleanup': 100000.0,
            'contingency_percentage': 15.0,
            'regulatory_filing_fees': 5000.0,
            'site_assessment_cost': 15000.0,
            'well_depth': 5000.0,
            'planned_abandonment_date': '2035-01-01'
        }

    if 'technical_parameters' not in st.session_state:
        st.session_state.technical_parameters = {
            'reservoir_pressure': 3000.0,
            'temperature': 180.0,
            'api_gravity': 35.0,
            'porosity': 20.0,
            'water_saturation': 30.0,
            'formation_volume_factor': 1.2,
            'gas_oil_ratio': 1000.0,
            'permeability': 50.0,
            'skin_factor': 0.0,
            'drainage_radius': 1000.0,
            'wellbore_radius': 0.328,
            'net_pay': 50.0
        }

    if 'regulatory_parameters' not in st.session_state:
        st.session_state.regulatory_parameters = {
            'last_inspection_date': '2024-01-01',
            'next_inspection_due': '2024-07-01',
            'permit_renewal_date': '2024-12-31',
            'compliance_score': 95.0,
            'environmental_risk': 'Low',
            'h2s_concentration': 0.0,
            'emission_rate': 0.0,
            'water_disposal_permit': 'Active',
            'spcc_plan_date': '2024-01-01',
            'safety_incidents': 0,
            'regulatory_violations': 0
        }

    if 'monte_carlo_parameters' not in st.session_state:
        st.session_state.monte_carlo_parameters = {
            'iterations': 1000,
            'confidence_level': 0.90,
            'seed': None,
            'price_volatility': 30.0,
            'production_uncertainty': 20.0,
            'cost_uncertainty': 15.0,
            'correlation_matrix': None,
            'scenario_weights': None
        }

    # Initialize calculation results and data storage
    if 'production_data' not in st.session_state:
        st.session_state.production_data = None

    if 'economic_results' not in st.session_state:
        st.session_state.economic_results = None

    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None

    if 'technical_results' not in st.session_state:
        st.session_state.technical_results = None

    if 'abandonment_results' not in st.session_state:
        st.session_state.abandonment_results = None


# def show_production_sidebar():
#     """Show sidebar inputs for production analysis."""
#     st.sidebar.header("Production Parameters")
#
#     params = st.session_state.production_parameters
#     params['initial_rate'] = st.sidebar.number_input(
#         "Initial Rate (bbl/day)",
#         value=params['initial_rate'],
#         min_value=0.0
#     )
#
#     params['decline_rate'] = st.sidebar.number_input(
#         "Annual Decline Rate (%)",
#         value=params['decline_rate'],
#         min_value=0.0,
#         max_value=100.0
#     )
#
#     params['forecast_months'] = st.sidebar.slider(
#         "Forecast Months",
#         min_value=12,
#         max_value=240,
#         value=params['forecast_months']
#     )
#
#     params['water_cut'] = st.sidebar.number_input(
#         "Current Water Cut (%)",
#         value=params['water_cut'],
#         min_value=0.0,
#         max_value=100.0
#     )
def show_production_sidebar():
    """Show sidebar inputs for production analysis."""
    params = get_production_params()
    st.session_state.production_parameters = params
# def show_economic_sidebar():
#     """Show sidebar inputs for economic analysis."""
#     st.sidebar.header("Economic Parameters")
#
#     params = st.session_state.economic_parameters
#     for key, (label, min_val, max_val) in {
#         'oil_price': ("Oil Price ($/bbl)", 0.0, None),
#         'opex': ("Operating Cost ($/bbl)", 0.0, None),
#         'initial_investment': ("Initial Investment ($)", 0.0, None),
#         'discount_rate': ("Discount Rate (%)", 0.0, 100.0),
#         'working_interest': ("Working Interest (%)", 0.0, 100.0),
#         'net_revenue_interest': ("Net Revenue Interest (%)", 0.0, 100.0)
#     }.items():
#         params[key] = st.sidebar.number_input(
#             label,
#             value=params[key],
#             min_value=min_val,
#             max_value=max_val
#         )
def show_economic_sidebar():
    """Show sidebar inputs for economic analysis."""
    params = get_economic_params()
    st.session_state.economic_parameters = params
def show_lease_sidebar():
    """Show sidebar inputs for lease analysis."""
    st.sidebar.header("Lease Parameters")

    params = st.session_state.lease_parameters
    for key, (label, min_val, max_val) in {
        'working_interest': ("Working Interest (%)", 0.0, 100.0),
        'net_revenue_interest': ("Net Revenue Interest (%)", 0.0, 100.0),
        'royalty_rate': ("Royalty Rate (%)", 0.0, 100.0),
        'lease_bonus': ("Lease Bonus ($)", 0.0, None),
        'lease_term_years': ("Lease Term (years)", 1.0, 50.0),
        'extension_cost': ("Extension Cost ($)", 0.0, None),
        'minimum_royalty': ("Minimum Royalty ($/month)", 0.0, None)
    }.items():
        params[key] = st.sidebar.number_input(
            label,
            value=params[key],
            min_value=min_val,
            max_value=max_val
        )
def show_abandonment_sidebar():
    """Show sidebar inputs for abandonment analysis."""
    st.sidebar.header("Abandonment Parameters")

    params = st.session_state.abandonment_parameters
    for key, (label, min_val, max_val) in {
        'plugging_cost_per_foot': ("Plugging Cost ($/ft)", 0.0, None),
        'site_restoration_cost': ("Site Restoration ($)", 0.0, None),
        'equipment_removal_cost': ("Equipment Removal ($)", 0.0, None),
        'environmental_cleanup': ("Environmental Cleanup ($)", 0.0, None),
        'contingency_percentage': ("Contingency (%)", 0.0, 100.0)
    }.items():
        params[key] = st.sidebar.number_input(
            label,
            value=params[key],
            min_value=min_val,
            max_value=max_val
        )
def show_technical_sidebar():
    """Show sidebar inputs for technical analysis."""
    st.sidebar.header("Technical Parameters")

    params = st.session_state.technical_parameters
    for key, (label, min_val, max_val) in {
        'reservoir_pressure': ("Reservoir Pressure (psi)", 0.0, None),
        'temperature': ("Temperature (°F)", 0.0, None),
        'api_gravity': ("API Gravity", 0.0, None),
        'porosity': ("Porosity (%)", 0.0, 100.0),
        'water_saturation': ("Water Saturation (%)", 0.0, 100.0),
        'formation_volume_factor': ("Formation Volume Factor", 1.0, None),
        'gas_oil_ratio': ("Gas-Oil Ratio (scf/bbl)", 0.0, None)
    }.items():
        params[key] = st.sidebar.number_input(
            label,
            value=params[key],
            min_value=min_val,
            max_value=max_val
        )
def show_regulatory_sidebar():
    """Show sidebar inputs for regulatory analysis."""
    st.sidebar.header("Regulatory Parameters")

    params = st.session_state.regulatory_parameters

    params['last_inspection_date'] = st.sidebar.date_input(
        "Last Inspection Date",
        value=datetime.strptime(params['last_inspection_date'], '%Y-%m-%d')
    ).strftime('%Y-%m-%d')

    params['next_inspection_due'] = st.sidebar.date_input(
        "Next Inspection Due",
        value=datetime.strptime(params['next_inspection_due'], '%Y-%m-%d')
    ).strftime('%Y-%m-%d')

    params['permit_renewal_date'] = st.sidebar.date_input(
        "Permit Renewal Date",
        value=datetime.strptime(params['permit_renewal_date'], '%Y-%m-%d')
    ).strftime('%Y-%m-%d')

    params['compliance_score'] = st.sidebar.slider(
        "Compliance Score",
        min_value=0.0,
        max_value=100.0,
        value=params['compliance_score']
    )

    params['environmental_risk'] = st.sidebar.selectbox(
        "Environmental Risk Level",
        options=['Low', 'Medium', 'High'],
        index=['Low', 'Medium', 'High'].index(params['environmental_risk'])
    )
def show_monte_carlo_sidebar():
    """Show sidebar inputs for Monte Carlo simulation."""
    st.sidebar.header("Monte Carlo Parameters")

    # Show economic parameters first
    show_economic_sidebar()

    # Add Monte Carlo specific parameters
    params = st.session_state.monte_carlo_parameters
    params['iterations'] = st.sidebar.number_input(
        "Number of Iterations",
        min_value=100,
        max_value=10000,
        value=params['iterations'],
        step=100
    )

    params['confidence_level'] = st.sidebar.slider(
        "Confidence Level",
        min_value=0.8,
        max_value=0.99,
        value=params['confidence_level'],
        step=0.01
    )



def update_session_parameters(section: str, params: dict):
    """Update session parameters while preserving existing values."""
    section_key = f"{section}_parameters"  # Make this consistent with our naming convention
    if section_key not in st.session_state:
        st.session_state[section_key] = {}
    st.session_state[section_key].update(params)
def get_well_params() -> dict:
    """Get well parameters from sidebar with unique keys."""
    params = {}
    current_page = st.session_state.get('current_page', 'Production Analysis')

    # Define well types options
    well_types = ['Oil', 'Gas', 'Dual']

    default_values = {
        'depth': ("Well Depth (ft)", 5000.0, 0.0),
        'age': ("Well Age (years)", 5.0, 0.0),
        'type': ("Well Type", well_types, None),
        'location': ("Location", "Default Field", None)
    }

    for param, (label, default_value, min_value) in default_values.items():
        # Create unique key combining page and parameter
        key = f"{current_page.lower().replace(' ', '_')}_well_{param}"

        # Initialize session state if not present
        if key not in st.session_state:
            st.session_state[key] = default_value if param != 'type' else well_types[0]

        if param == 'type':
            params[param] = st.sidebar.selectbox(
                label,
                options=well_types,
                key=key
            )
        elif param == 'location':
            params[param] = st.sidebar.text_input(
                label,
                value=st.session_state[key],
                key=key
            )
        else:
            params[param] = st.sidebar.number_input(
                label,
                min_value=min_value,
                value=st.session_state[key],
                key=key
            )

    return params


def create_page_sidebar(tab_name: str, prefix: str):
    """Create sidebar based on current page with unique widget keys."""
    st.sidebar.header("Input Parameters")

    # Well Information (common across all tabs)
    st.sidebar.subheader("Well Information")
    well_params = st.session_state.well_parameters

    well_params['depth'] = st.sidebar.number_input(
        "Well Depth (ft)",
        value=well_params['depth'],
        min_value=0.0,
        key=f"{prefix}_well_depth"
    )

    well_params['age'] = st.sidebar.number_input(
        "Well Age (years)",
        value=well_params['age'],
        min_value=0.0,
        key=f"{prefix}_well_age"
    )

    well_params['type'] = st.sidebar.selectbox(
        "Well Type",
        options=['Oil', 'Gas', 'Dual'],
        index=['Oil', 'Gas', 'Dual'].index(well_params['type']),
        key=f"{prefix}_well_type"
    )

    # Tab-specific parameters
    if tab_name == "Production Analysis":
        st.sidebar.subheader("Production Parameters")
        params = get_production_params(prefix)
        st.session_state.production_parameters.update(params)

    elif tab_name == "Economic Analysis":
        st.sidebar.subheader("Economic Parameters")
        params = get_economic_params(prefix)
        st.session_state.economic_parameters.update(params)

    elif tab_name == "Monte Carlo Simulation":
        st.sidebar.subheader("Monte Carlo Parameters")
        params = get_monte_carlo_params(prefix)
        st.session_state.monte_carlo_parameters.update(params)

    elif tab_name == "Technical Analysis":
        st.sidebar.subheader("Technical Parameters")
        params = get_technical_params(prefix)
        st.session_state.technical_parameters.update(params)

    elif tab_name == "Lease Analysis":
        st.sidebar.subheader("Lease Parameters")
        params = get_lease_params(prefix)
        st.session_state.lease_parameters.update(params)

    elif tab_name == "Abandonment Analysis":
        st.sidebar.subheader("Abandonment Parameters")
        params = get_abandonment_params(prefix)
        st.session_state.abandonment_parameters.update(params)

    elif tab_name == "Regulatory Compliance":
        st.sidebar.subheader("Regulatory Parameters")
        params = get_regulatory_params(prefix)
        st.session_state.regulatory_parameters.update(params)
def get_production_params(prefix: str) -> dict:
    """Get production parameters with unique keys."""
    params = st.session_state.production_parameters

    updated_params = {}

    updated_params['initial_rate'] = st.sidebar.number_input(
        "Initial Rate (bbl/day)",
        value=params['initial_rate'],
        min_value=0.0,
        key=f"{prefix}_initial_rate"
    )

    updated_params['decline_rate'] = st.sidebar.number_input(
        "Annual Decline Rate (%)",
        value=params['decline_rate'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_decline_rate"
    )

    updated_params['forecast_months'] = st.sidebar.slider(
        "Forecast Months",
        min_value=12,
        max_value=240,
        value=params['forecast_months'],
        key=f"{prefix}_forecast_months"
    )

    updated_params['water_cut'] = st.sidebar.number_input(
        "Water Cut (%)",
        value=params['water_cut'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_water_cut"
    )

    updated_params['gas_oil_ratio'] = st.sidebar.number_input(
        "Gas-Oil Ratio (scf/bbl)",
        value=params['gas_oil_ratio'],
        min_value=0.0,
        key=f"{prefix}_gor"
    )

    return updated_params
def get_economic_params(prefix: str) -> dict:
    """Get economic parameters with unique keys."""
    params = st.session_state.economic_parameters

    updated_params = {}

    updated_params['oil_price'] = st.sidebar.number_input(
        "Oil Price ($/bbl)",
        value=params['oil_price'],
        min_value=0.0,
        key=f"{prefix}_oil_price"
    )

    updated_params['gas_price'] = st.sidebar.number_input(
        "Gas Price ($/mcf)",
        value=params['gas_price'],
        min_value=0.0,
        key=f"{prefix}_gas_price"
    )

    updated_params['opex'] = st.sidebar.number_input(
        "Operating Cost ($/bbl)",
        value=params['opex'],
        min_value=0.0,
        key=f"{prefix}_opex"
    )

    updated_params['initial_investment'] = st.sidebar.number_input(
        "Initial Investment ($)",
        value=params['initial_investment'],
        min_value=0.0,
        key=f"{prefix}_initial_investment"
    )

    updated_params['discount_rate'] = st.sidebar.number_input(
        "Discount Rate (%)",
        value=params['discount_rate'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_discount_rate"
    )

    updated_params['working_interest'] = st.sidebar.number_input(
        "Working Interest (%)",
        value=params['working_interest'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_working_interest"
    )

    updated_params['net_revenue_interest'] = st.sidebar.number_input(
        "Net Revenue Interest (%)",
        value=params['net_revenue_interest'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_net_revenue_interest"
    )

    return updated_params
def get_monte_carlo_params(prefix: str) -> dict:
    """Get Monte Carlo parameters with unique keys."""
    params = st.session_state.monte_carlo_parameters

    updated_params = {}

    updated_params['iterations'] = st.sidebar.number_input(
        "Number of Iterations",
        value=params['iterations'],
        min_value=100,
        max_value=10000,
        step=100,
        key=f"{prefix}_iterations"
    )

    updated_params['confidence_level'] = st.sidebar.slider(
        "Confidence Level",
        min_value=0.80,
        max_value=0.99,
        value=params['confidence_level'],
        step=0.01,
        key=f"{prefix}_confidence"
    )

    updated_params['price_volatility'] = st.sidebar.slider(
        "Price Volatility (%)",
        min_value=0.0,
        max_value=100.0,
        value=params['price_volatility'],
        key=f"{prefix}_price_volatility"
    )

    updated_params['production_uncertainty'] = st.sidebar.slider(
        "Production Uncertainty (%)",
        min_value=0.0,
        max_value=100.0,
        value=params['production_uncertainty'],
        key=f"{prefix}_prod_uncertainty"
    )

    return updated_params


def get_technical_params(prefix: str) -> dict:
    """Get technical parameters with unique keys."""
    params = st.session_state.technical_parameters

    updated_params = {}

    updated_params['reservoir_pressure'] = st.sidebar.number_input(
        "Reservoir Pressure (psi)",
        value=params['reservoir_pressure'],
        min_value=0.0,
        key=f"{prefix}_reservoir_pressure"
    )

    updated_params['temperature'] = st.sidebar.number_input(
        "Temperature (°F)",
        value=params['temperature'],
        min_value=0.0,
        key=f"{prefix}_temperature"
    )

    updated_params['api_gravity'] = st.sidebar.number_input(
        "API Gravity",
        value=params['api_gravity'],
        min_value=0.0,
        key=f"{prefix}_api_gravity"
    )

    updated_params['porosity'] = st.sidebar.number_input(
        "Porosity (%)",
        value=params['porosity'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_porosity"
    )

    updated_params['water_saturation'] = st.sidebar.number_input(
        "Water Saturation (%)",
        value=params['water_saturation'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_water_saturation"
    )

    updated_params['formation_volume_factor'] = st.sidebar.number_input(
        "Formation Volume Factor",
        value=params['formation_volume_factor'],
        min_value=1.0,
        key=f"{prefix}_formation_volume_factor"
    )

    updated_params['gas_oil_ratio'] = st.sidebar.number_input(
        "Gas-Oil Ratio (scf/bbl)",
        value=params['gas_oil_ratio'],
        min_value=0.0,
        key=f"{prefix}_tech_gor"  # Changed to avoid conflict with production GOR
    )

    updated_params['permeability'] = st.sidebar.number_input(
        "Permeability (md)",
        value=params['permeability'],
        min_value=0.0,
        key=f"{prefix}_permeability"
    )

    updated_params['skin_factor'] = st.sidebar.number_input(
        "Skin Factor",
        value=params['skin_factor'],
        key=f"{prefix}_skin_factor"
    )

    updated_params['drainage_radius'] = st.sidebar.number_input(
        "Drainage Radius (ft)",
        value=params['drainage_radius'],
        min_value=0.0,
        key=f"{prefix}_drainage_radius"
    )

    updated_params['wellbore_radius'] = st.sidebar.number_input(
        "Wellbore Radius (ft)",
        value=params['wellbore_radius'],
        min_value=0.0,
        key=f"{prefix}_wellbore_radius"
    )

    updated_params['net_pay'] = st.sidebar.number_input(
        "Net Pay (ft)",
        value=params['net_pay'],
        min_value=0.0,
        key=f"{prefix}_net_pay"
    )

    return updated_params

def get_lease_params(prefix: str) -> dict:
    """Get lease parameters with unique keys."""
    params = st.session_state.lease_parameters

    updated_params = {}

    updated_params['working_interest'] = st.sidebar.number_input(
        "Working Interest (%)",
        value=params['working_interest'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_working_interest"
    )

    updated_params['net_revenue_interest'] = st.sidebar.number_input(
        "Net Revenue Interest (%)",
        value=params['net_revenue_interest'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_net_revenue_interest"
    )

    updated_params['royalty_rate'] = st.sidebar.number_input(
        "Royalty Rate (%)",
        value=params['royalty_rate'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_royalty_rate"
    )

    updated_params['lease_bonus'] = st.sidebar.number_input(
        "Lease Bonus ($)",
        value=params['lease_bonus'],
        min_value=0.0,
        key=f"{prefix}_lease_bonus"
    )

    updated_params['lease_term_years'] = st.sidebar.number_input(
        "Lease Term (years)",
        value=params['lease_term_years'],
        min_value=1.0,
        max_value=50.0,
        key=f"{prefix}_lease_term"
    )

    updated_params['extension_cost'] = st.sidebar.number_input(
        "Extension Cost ($)",
        value=params['extension_cost'],
        min_value=0.0,
        key=f"{prefix}_extension_cost"
    )

    updated_params['minimum_royalty'] = st.sidebar.number_input(
        "Minimum Royalty ($/month)",
        value=params['minimum_royalty'],
        min_value=0.0,
        key=f"{prefix}_minimum_royalty"
    )

    updated_params['delay_rentals'] = st.sidebar.number_input(
        "Delay Rentals ($/year)",
        value=params['delay_rentals'],
        min_value=0.0,
        key=f"{prefix}_delay_rentals"
    )

    # Date inputs
    updated_params['primary_term_end'] = st.sidebar.date_input(
        "Primary Term End Date",
        value=datetime.strptime(params['primary_term_end'], '%Y-%m-%d'),
        key=f"{prefix}_primary_term_end"
    ).strftime('%Y-%m-%d')

    updated_params['lease_expiration'] = st.sidebar.date_input(
        "Lease Expiration Date",
        value=datetime.strptime(params['lease_expiration'], '%Y-%m-%d'),
        key=f"{prefix}_lease_expiration"
    ).strftime('%Y-%m-%d')

    return updated_params


def get_abandonment_params(prefix: str) -> dict:
    """Get abandonment parameters with unique keys."""
    params = st.session_state.abandonment_parameters

    updated_params = {}

    updated_params['plugging_cost_per_foot'] = st.sidebar.number_input(
        "Plugging Cost ($/ft)",
        value=params['plugging_cost_per_foot'],
        min_value=0.0,
        key=f"{prefix}_plugging_cost"
    )

    updated_params['site_restoration_cost'] = st.sidebar.number_input(
        "Site Restoration ($)",
        value=params['site_restoration_cost'],
        min_value=0.0,
        key=f"{prefix}_site_restoration"
    )

    updated_params['equipment_removal_cost'] = st.sidebar.number_input(
        "Equipment Removal ($)",
        value=params['equipment_removal_cost'],
        min_value=0.0,
        key=f"{prefix}_equipment_removal"
    )

    updated_params['environmental_cleanup'] = st.sidebar.number_input(
        "Environmental Cleanup ($)",
        value=params['environmental_cleanup'],
        min_value=0.0,
        key=f"{prefix}_environmental"
    )

    updated_params['contingency_percentage'] = st.sidebar.number_input(
        "Contingency (%)",
        value=params['contingency_percentage'],
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_contingency"
    )

    updated_params['regulatory_filing_fees'] = st.sidebar.number_input(
        "Regulatory Filing Fees ($)",
        value=params['regulatory_filing_fees'],
        min_value=0.0,
        key=f"{prefix}_regulatory_fees"
    )

    updated_params['site_assessment_cost'] = st.sidebar.number_input(
        "Site Assessment ($)",
        value=params['site_assessment_cost'],
        min_value=0.0,
        key=f"{prefix}_assessment_cost"
    )

    updated_params['well_depth'] = st.sidebar.number_input(
        "Abandonment Depth (ft)",  # Changed label to differentiate
        value=params['well_depth'],
        min_value=0.0,
        key=f"{prefix}_abandonment_depth"  # Changed key to avoid duplication
    )

    # Date input
    updated_params['planned_abandonment_date'] = st.sidebar.date_input(
        "Planned Abandonment Date",
        value=datetime.strptime(params['planned_abandonment_date'], '%Y-%m-%d'),
        key=f"{prefix}_planned_date"
    ).strftime('%Y-%m-%d')

    return updated_params


def get_regulatory_params(prefix: str) -> dict:
    """Get regulatory parameters with unique keys."""
    # Get current values from session state, ensuring all fields exist
    params = st.session_state.regulatory_parameters

    updated_params = {}

    # Regulatory specific parameters first (no well info overlap)
    updated_params['compliance_score'] = st.sidebar.number_input(
        "Compliance Score",
        value=params.get('compliance_score', 95.0),
        min_value=0.0,
        max_value=100.0,
        key=f"{prefix}_compliance_score"
    )

    updated_params['h2s_concentration'] = st.sidebar.number_input(
        "H2S Concentration (ppm)",
        value=params.get('h2s_concentration', 0.0),
        min_value=0.0,
        key=f"{prefix}_h2s_concentration"
    )

    updated_params['emission_rate'] = st.sidebar.number_input(
        "Emission Rate (mcf/d)",
        value=params.get('emission_rate', 0.0),
        min_value=0.0,
        key=f"{prefix}_emission_rate"
    )

    updated_params['environmental_risk'] = st.sidebar.selectbox(
        "Environmental Risk Level",
        options=['Low', 'Medium', 'High'],
        index=['Low', 'Medium', 'High'].index(params.get('environmental_risk', 'Low')),
        key=f"{prefix}_environmental_risk"
    )

    updated_params['water_disposal_permit'] = st.sidebar.selectbox(
        "Water Disposal Permit Status",
        options=['Active', 'Pending', 'Expired'],
        index=['Active', 'Pending', 'Expired'].index(params.get('water_disposal_permit', 'Active')),
        key=f"{prefix}_water_permit"
    )

    # Dates
    try:
        last_inspection = datetime.strptime(params.get('last_inspection_date', '2024-01-01'), '%Y-%m-%d')
    except (ValueError, TypeError):
        last_inspection = datetime.now()

    updated_params['last_inspection_date'] = st.sidebar.date_input(
        "Last Inspection Date",
        value=last_inspection,
        key=f"{prefix}_last_inspection"
    ).strftime('%Y-%m-%d')

    try:
        next_inspection = datetime.strptime(params.get('next_inspection_due', '2024-07-01'), '%Y-%m-%d')
    except (ValueError, TypeError):
        next_inspection = datetime.now()

    updated_params['next_inspection_due'] = st.sidebar.date_input(
        "Next Inspection Due",
        value=next_inspection,
        key=f"{prefix}_next_inspection"
    ).strftime('%Y-%m-%d')

    try:
        permit_renewal = datetime.strptime(params.get('permit_renewal_date', '2024-12-31'), '%Y-%m-%d')
    except (ValueError, TypeError):
        permit_renewal = datetime.now()

    updated_params['permit_renewal_date'] = st.sidebar.date_input(
        "Permit Renewal Date",
        value=permit_renewal,
        key=f"{prefix}_permit_renewal"
    ).strftime('%Y-%m-%d')

    try:
        spcc_date = datetime.strptime(params.get('spcc_plan_date', '2024-01-01'), '%Y-%m-%d')
    except (ValueError, TypeError):
        spcc_date = datetime.now()

    updated_params['spcc_plan_date'] = st.sidebar.date_input(
        "SPCC Plan Date",
        value=spcc_date,
        key=f"{prefix}_spcc_date"
    ).strftime('%Y-%m-%d')

    # Integer inputs
    updated_params['safety_incidents'] = st.sidebar.number_input(
        "Safety Incidents",
        value=params.get('safety_incidents', 0),
        min_value=0,
        step=1,
        key=f"{prefix}_safety_incidents"
    )

    updated_params['regulatory_violations'] = st.sidebar.number_input(
        "Regulatory Violations",
        value=params.get('regulatory_violations', 0),
        min_value=0,
        step=1,
        key=f"{prefix}_violations"
    )

    return updated_params

def calculate_technical_parameters(
    pressure: float,
    temperature: float,
    api_gravity: float,
    porosity: float = 0.20,
    water_saturation: float = 0.30,
    formation_thickness: float = 50.0,
) -> Dict[str, float]:
    """Calculate technical parameters for the well."""
    # Basic PVT correlations
    gas_gravity = 0.65  # assumed
    rsb = 0.0362 * gas_gravity * pressure * (10 ** (0.0125 * api_gravity)) * \
          (temperature + 460) ** (-1.2)

    # Calculate oil formation volume factor
    bob = 1 + 5.615 * (rsb * gas_gravity / (api_gravity + 131.5)) * \
          (temperature + 460) / 5.615

    # Calculate oil viscosity
    dead_oil_visc = 10 ** (0.43 + (8.33 / api_gravity))
    oil_visc = dead_oil_visc * (1 + 0.001 * rsb) ** 0.2

    # Calculate hydrocarbon pore volume
    hc_pore_volume = porosity * (1 - water_saturation) * formation_thickness

    # Calculate original oil in place (OOIP)
    ooip = 7758 * porosity * (1 - water_saturation) * formation_thickness / bob

    return {
        'solution_gas': rsb,
        'formation_volume_factor': bob,
        'oil_viscosity': oil_visc,
        'hydrocarbon_pore_volume': hc_pore_volume,
        'original_oil_in_place': ooip,
        'gas_oil_ratio': rsb,  # Initial GOR equals solution gas ratio
        'effective_porosity': porosity * (1 - water_saturation)
    }


def run_production_analysis():
    """Run production analysis with current parameters."""
    st.header("Production Analysis")

    try:
        # Get current parameters
        params = st.session_state.production_parameters

        # Convert percentages to decimals for calculations
        decline_rate = params['decline_rate'] / 100
        water_cut = params['water_cut'] / 100

        # Calculate production profile
        months = params['forecast_months']
        time = np.arange(months)
        production = params['initial_rate'] * np.exp(-decline_rate * time / 12)

        # Calculate water cut progression
        water_cut_profile = np.minimum(water_cut * np.exp(0.01 * time), 0.95)

        # Create production dataframe
        prod_df = pd.DataFrame({
            'Month': time,
            'Production': production,
            'Water_Cut': water_cut_profile,
            'Oil_Production': production * (1 - water_cut_profile),
            'Water_Production': production * water_cut_profile
        })

        # Store results in session state
        st.session_state.production_data = prod_df

        # Display production plot
        fig = create_production_plot(prod_df)
        st.plotly_chart(fig, use_container_width=True)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cumulative Oil Production",
                      f"{(production * (1 - water_cut_profile)).sum():,.0f} bbl")
            st.metric("Average Oil Rate",
                      f"{(production * (1 - water_cut_profile)).mean():,.0f} bbl/d")
        with col2:
            st.metric("Initial Rate", f"{params['initial_rate']:,.0f} bbl/d")
            st.metric("Final Rate", f"{production[-1]:,.0f} bbl/d")
        with col3:
            st.metric("Decline Rate", f"{params['decline_rate']:.1f}%/year")
            st.metric("Current Water Cut", f"{params['water_cut']:.1f}%")

    except Exception as e:
        st.error(f"Error in production analysis: {str(e)}")
        st.exception(e)
def run_economic_analysis():
    """Run economic analysis with current parameters."""
    st.header("Economic Analysis")

    try:
        # Get current parameters
        econ_params = st.session_state.economic_parameters
        prod_data = st.session_state.production_data

        if prod_data is None:
            st.error("Please run Production Analysis first")
            return

        # Convert percentages to decimals
        discount_rate = econ_params['discount_rate'] / 100
        working_interest = econ_params['working_interest'] / 100
        net_revenue_interest = econ_params['net_revenue_interest'] / 100

        # Calculate revenue and costs
        oil_production = prod_data['Production'] * (1 - prod_data['Water_Cut'])
        gross_revenue = oil_production * econ_params['oil_price']
        operating_costs = prod_data['Production'] * econ_params['opex']

        # Calculate net revenue
        net_revenue = (gross_revenue * net_revenue_interest -
                       operating_costs * working_interest)

        # Calculate NPV
        time = np.arange(len(prod_data))
        monthly_discount_rate = (1 + discount_rate) ** (1 / 12) - 1
        discount_factors = 1 / (1 + monthly_discount_rate) ** time
        npv = -econ_params['initial_investment'] + np.sum(net_revenue * discount_factors)

        # Calculate other metrics
        total_revenue = gross_revenue.sum()
        total_costs = operating_costs.sum() + econ_params['initial_investment']
        net_profit = total_revenue - total_costs
        roi = (net_profit / econ_params['initial_investment']) * 100 if econ_params['initial_investment'] > 0 else 0

        # Store results
        st.session_state.economic_results = {
            'npv': npv,
            'roi': roi,
            'total_revenue': total_revenue,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'monthly_net_revenue': net_revenue
        }

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Net Present Value", f"${npv:,.0f}")
            st.metric("ROI", f"{roi:.1f}%")
        with col2:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
            st.metric("Total Costs", f"${total_costs:,.0f}")
        with col3:
            st.metric("Net Profit", f"${net_profit:,.0f}")
            st.metric("Average Monthly Revenue", f"${net_revenue.mean():,.0f}")

        # Create and display cash flow plot
        cash_flow_df = pd.DataFrame({
            'Month': prod_data['Month'],
            'Net_Revenue': net_revenue,
            'Cumulative_Cash_Flow': np.cumsum(net_revenue)
        })

        fig = create_cash_flow_plot(cash_flow_df)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in economic analysis: {str(e)}")
        st.exception(e)
def run_monte_carlo_analysis():
    """Run Monte Carlo analysis tab content."""
    st.header("Monte Carlo Simulation")

    mc_params = st.session_state.monte_carlo_parameters
    econ_params = st.session_state.economic_parameters
    prod_params = st.session_state.production_parameters

    # Create EconomicParameters instance
    try:
        economic_params = EconomicParameters(
            oil_price=econ_params['oil_price'],
            opex=econ_params['opex'],
            initial_investment=econ_params['initial_investment'],
            discount_rate=econ_params['discount_rate'],
            initial_rate=prod_params['initial_rate'],
            decline_rate=prod_params['decline_rate'],
            working_interest=econ_params['working_interest'],
            net_revenue_interest=econ_params['net_revenue_interest'],
            lease_terms=econ_params['lease_terms'],
            abandonment_costs=econ_params['abandonment_costs']
        )
    except Exception as e:
        st.error(f"Error creating economic parameters: {str(e)}")
        return

    # Run simulation
    try:
        with st.spinner("Running Monte Carlo simulation..."):
            results = st.session_state.monte_carlo.run_full_analysis(
                economic_params=economic_params,
                months=prod_params['forecast_months'],
                iterations=mc_params['iterations'],
                confidence_level=mc_params['confidence_level']
            )
    except Exception as e:
        st.error(f"Error running Monte Carlo simulation: {str(e)}")
        return

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Production Statistics")
        prod_stats = pd.DataFrame({
            'Metric': ['P10', 'P50', 'P90', 'Mean', 'Std Dev'],
            'Value (bbl/d)': [
                f"{results.percentiles['Production']['P10']:,.0f}",
                f"{results.percentiles['Production']['P50']:,.0f}",
                f"{results.percentiles['Production']['P90']:,.0f}",
                f"{results.statistics['Production']['mean']:,.0f}",
                f"{results.statistics['Production']['std']:,.0f}"
            ]
        })
        st.dataframe(prod_stats)

    with col2:
        st.subheader("NPV Statistics")
        npv_stats = pd.DataFrame({
            'Metric': ['P10', 'P50', 'P90', 'Mean', 'Std Dev'],
            'Value ($)': [
                f"${results.percentiles['NPV']['P10']:,.0f}",
                f"${results.percentiles['NPV']['P50']:,.0f}",
                f"${results.percentiles['NPV']['P90']:,.0f}",
                f"${results.statistics['NPV']['mean']:,.0f}",
                f"${results.statistics['NPV']['std']:,.0f}"
            ]
        })
        st.dataframe(npv_stats)

    with col3:
        st.subheader("Risk Metrics")
        risk_metrics = pd.DataFrame({
            'Metric': [
                'Probability of Loss',
                'Value at Risk (95%)',
                'Expected Shortfall',
                'Probability of Target ROI'
            ],
            'Value': [
                f"{results.risk_metrics['probability_of_loss'] * 100:.1f}%",
                f"${results.risk_metrics['value_at_risk']:,.0f}",
                f"${results.risk_metrics['expected_shortfall']:,.0f}",
                f"{results.risk_metrics['probability_of_target_roi'] * 100:.1f}%"
            ]
        })
        st.dataframe(risk_metrics)

    # Monte Carlo plots
    st.subheader("Simulation Results")
    figs = create_monte_carlo_plots(results)
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

    # Risk analysis plots
    st.subheader("Risk Analysis")
    fig_risk = create_risk_metrics_plot(results)
    st.plotly_chart(fig_risk, use_container_width=True)

    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    sensitivity_ranges = {
        'Oil Price': (econ_params['oil_price'] * 0.7,
                      econ_params['oil_price'] * 1.3),
        'OPEX': (econ_params['opex'] * 0.8,
                 econ_params['opex'] * 1.2),
        'Initial Rate': (prod_params['initial_rate'] * 0.9,
                         prod_params['initial_rate'] * 1.1),
        'Decline Rate': (prod_params['decline_rate'] * 0.8,
                         prod_params['decline_rate'] * 1.2)
    }
    fig_tornado = create_tornado_plot(results, results.statistics['NPV']['mean'], sensitivity_ranges)
    st.plotly_chart(fig_tornado, use_container_width=True)
def run_lease_analysis():
    """Run lease analysis tab content."""
    st.header("Lease Analysis")

    lease_params = st.session_state.lease_parameters
    prod_data = st.session_state.production_data
    econ_params = st.session_state.economic_parameters

    if prod_data is None:
        st.error("Please run Production Analysis first")
        return

    # Calculate lease metrics
    lease_terms = LeaseTerms(
        working_interest=lease_params['working_interest'],
        net_revenue_interest=lease_params['net_revenue_interest'],
        royalty_rate=lease_params['royalty_rate'],
        lease_bonus=lease_params['lease_bonus'],
        lease_term_years=lease_params['lease_term_years'],
        extension_cost=lease_params['extension_cost'],
        minimum_royalty=lease_params['minimum_royalty']
    )

    lease_results = st.session_state.lease.calculate_lease_economics(
        production=prod_data['Production'].values,
        oil_price=econ_params['oil_price'],
        lease_terms=lease_terms,
        current_month=0
    )

    # Display lease metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Working Interest Revenue",
            f"${lease_results['working_interest_revenue']:,.0f}"
        )
        st.metric(
            "Net Revenue Interest",
            f"${lease_results['net_revenue']:,.0f}"
        )
    with col2:
        st.metric(
            "Total Royalties",
            f"${lease_results['total_royalties']:,.0f}"
        )
        st.metric(
            "Extension Cost",
            f"${lease_results['extension_cost']:,.0f}"
        )

    # Display lease status
    st.subheader("Lease Status")
    if lease_results['remaining_term_months'] <= 0:
        st.warning(f"⚠️ Lease has expired. Extension cost: ${lease_results['extension_cost']:,.2f}")
    else:
        st.success(f"✅ Lease active. {lease_results['remaining_term_months']} months remaining")
def run_abandonment_analysis():
    """Run abandonment analysis tab content."""
    st.header("Abandonment Analysis")

    aband_params = st.session_state.abandonment_parameters
    well_params = st.session_state.well_parameters
    econ_params = st.session_state.economic_parameters
    prod_data = st.session_state.production_data

    if prod_data is None:
        st.error("Please run Production Analysis first")
        return

    # Calculate abandonment costs
    abandonment_costs = AbandonmentCosts(
        plugging_cost=well_params['depth'] * aband_params['plugging_cost_per_foot'],
        site_restoration=aband_params['site_restoration_cost'],
        equipment_removal=aband_params['equipment_removal_cost'],
        environmental_cleanup=aband_params['environmental_cleanup'],
        regulatory_fees=aband_params['regulatory_filing_fees'],
        contingency=aband_params['contingency_percentage']
    )

    # Calculate total obligation
    obligation = st.session_state.lease.calculate_total_abandonment_obligation(
        costs=abandonment_costs,
        inflation_years=well_params['age']
    )

    # Display abandonment costs
    st.subheader("Abandonment Cost Breakdown")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Total Abandonment Cost",
            f"${obligation['total_obligation']:,.0f}",
            delta=f"+{obligation['inflation_adjustment']:,.0f} inflation adjustment"
        )
        st.metric(
            "Base Cost",
            f"${obligation['base_costs']:,.0f}"
        )
    with col2:
        st.metric(
            "Contingency",
            f"${obligation['contingency_amount']:,.0f}"
        )
        st.metric(
            "Per Foot Cost",
            f"${obligation['total_obligation'] / well_params['depth']:,.2f}/ft"
        )

    # Detailed cost breakdown
    st.subheader("Cost Components")
    cost_breakdown = pd.DataFrame({
        'Category': list(obligation['cost_breakdown'].keys()),
        'Amount': list(obligation['cost_breakdown'].values())
    })
    cost_breakdown['Percentage'] = cost_breakdown['Amount'] / cost_breakdown['Amount'].sum() * 100

    # Display cost breakdown chart
    fig = go.Figure(data=[go.Pie(
        labels=cost_breakdown['Category'],
        values=cost_breakdown['Amount'],
        hole=.3
    )])
    fig.update_layout(title="Cost Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Abandonment timing analysis
    st.subheader("Abandonment Timing Analysis")

    # Calculate economic limit
    economic_limit = econ_params['opex'] * 1.5  # 1.5x operating cost
    production = prod_data['Production'].values
    revenue = production * econ_params['oil_price']

    months_to_limit = np.where(revenue < economic_limit * production)[0]
    if len(months_to_limit) > 0:
        economic_limit_date = datetime.now() + timedelta(days=30 * months_to_limit[0])
        st.warning(f"⚠️ Economic limit reached in {months_to_limit[0]} months "
                   f"(approximately {economic_limit_date.strftime('%B %Y')})")
    else:
        st.success("✅ Well remains economic throughout forecast period")

    # Planned abandonment date
    st.subheader("Planned Abandonment")
    planned_date = datetime.strptime(aband_params['planned_abandonment_date'], '%Y-%m-%d')
    months_to_planned = (planned_date - datetime.now()).days / 30

    st.info(f"🗓️ Planned abandonment date: {planned_date.strftime('%B %d, %Y')} "
            f"({months_to_planned:.1f} months from now)")
def run_technical_analysis():
    """Run technical analysis tab content."""
    st.header("Technical Analysis")

    tech_params = st.session_state.technical_parameters

    # Calculate technical parameters
    tech_results = calculate_technical_parameters(
        pressure=tech_params['reservoir_pressure'],
        temperature=tech_params['temperature'],
        api_gravity=tech_params['api_gravity']
    )

    # Display reservoir parameters
    st.subheader("Reservoir Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Reservoir Pressure", f"{tech_params['reservoir_pressure']:,.0f} psi")
        st.metric("Formation Volume Factor", f"{tech_results['formation_volume_factor']:.3f} RB/STB")

    with col2:
        st.metric("Temperature", f"{tech_params['temperature']}°F")
        st.metric("Oil Viscosity", f"{tech_results['oil_viscosity']:.2f} cp")

    with col3:
        st.metric("API Gravity", f"{tech_params['api_gravity']:.1f}°API")
        st.metric("Solution GOR", f"{tech_results['solution_gas']:,.0f} scf/STB")

    # Additional reservoir properties
    st.subheader("Additional Properties")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Porosity", f"{tech_params['porosity'] * 100:.1f}%")
        st.metric("Water Saturation", f"{tech_params['water_saturation'] * 100:.1f}%")

    with col2:
        st.metric("Permeability", f"{tech_params['permeability']:.1f} md")
        st.metric("Net Pay", f"{tech_params['net_pay']:.1f} ft")

    with col3:
        st.metric("Gas-Oil Ratio", f"{tech_params['gas_oil_ratio']:,.0f} scf/bbl")
        st.metric("Formation Volume Factor", f"{tech_params['formation_volume_factor']:.3f} rb/stb")

    # Technical plots
    st.subheader("Technical Analysis")
    figs = create_technical_plots(tech_params)
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)
def run_regulatory_analysis():
    """Run regulatory analysis tab content."""
    st.header("Regulatory Compliance")

    reg_params = st.session_state.regulatory_parameters

    # Compliance Status
    st.subheader("Compliance Status")

    # Environmental Compliance - using actual stored parameters
    compliance_categories = {
        'Air Emissions': {
            'status': 'Compliant' if reg_params['emission_rate'] < 100 else 'Warning',
            'last_inspection': reg_params['last_inspection_date'],
            'next_due': reg_params['next_inspection_due']
        },
        'Water Disposal': {
            'status': reg_params['water_disposal_permit'],
            'last_inspection': reg_params['last_inspection_date'],
            'next_due': reg_params['next_inspection_due']
        },
        'Spill Prevention': {
            'status': 'Compliant' if reg_params['spcc_plan_date'] > datetime.now().strftime('%Y-%m-%d') else 'Warning',
            'last_inspection': reg_params['spcc_plan_date'],
            'next_due': datetime.strptime(reg_params['spcc_plan_date'], '%Y-%m-%d') + timedelta(days=365)
        },
        'Safety Compliance': {
            'status': 'Compliant' if reg_params['safety_incidents'] == 0 else 'Action Required',
            'incidents': reg_params['safety_incidents'],
            'risk_level': reg_params['environmental_risk']
        }
    }

    for category, details in compliance_categories.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if details['status'] == 'Compliant':
                st.success(f"{category}: {details['status']}")
            elif details['status'] == 'Warning':
                st.warning(f"{category}: {details['status']}")
            else:
                st.error(f"{category}: {details['status']}")
        with col2:
            if 'last_inspection' in details:
                st.write(f"Last: {details['last_inspection']}")
        with col3:
            if 'next_due' in details:
                st.write(f"Next: {details['next_due']}")

    # Overall Compliance Score
    st.subheader("Overall Compliance")
    st.progress(reg_params['compliance_score'] / 100)
    st.metric("Compliance Score", f"{reg_params['compliance_score']:.1f}%")

    # Permit Status
    st.subheader("Permit Status")
    permits = pd.DataFrame({
        'Permit Type': ['Operating Permit', 'Water Disposal', 'Air Quality', 'Land Use'],
        'Status': ['Active', reg_params['water_disposal_permit'], 'Renewal Required', 'Active'],
        'Expiration Date': [
            reg_params['permit_renewal_date'],
            reg_params['next_inspection_due'],
            reg_params['permit_renewal_date'],
            reg_params['permit_renewal_date']
        ],
        'Renewal Cost': [5000, 3000, 2500, 1500]
    })

    # Apply color coding to status
    def color_status(val):
        if val == 'Active':
            return 'background-color: #90EE90'
        elif val in ['Renewal Required', 'Warning']:
            return 'background-color: #FFB6C1'
        return ''

    st.dataframe(permits.style.applymap(color_status, subset=['Status']))

    # Additional Metrics
    st.subheader("Environmental Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("H2S Concentration", f"{reg_params['h2s_concentration']:.1f} ppm")
        st.metric("Emission Rate", f"{reg_params['emission_rate']:.1f} mcf/d")

    with col2:
        st.metric("Safety Incidents", str(reg_params['safety_incidents']))
        st.metric("Regulatory Violations", str(reg_params['regulatory_violations']))


def run_lease_analysis():
    """Run lease analysis tab content."""
    st.header("Lease Analysis")

    # Get parameters from session state
    lease_params = st.session_state.lease_parameters
    econ_params = st.session_state.economic_parameters

    # Investment Analysis section
    st.subheader("Investment Analysis")

    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        initial_investment = st.number_input(
            "Initial Investment ($)",
            value=5000000.0,
            min_value=0.0,
            step=100000.0,
            format="%0.2f"
        )

        oil_price = st.number_input(
            "Oil Price ($/bbl)",
            value=70.0,
            min_value=0.0,
            step=1.0
        )

    with col2:
        operating_cost = st.number_input(
            "Operating Cost ($/bbl)",
            value=25.0,
            min_value=0.0,
            step=1.0
        )

        decline_rate = st.number_input(
            "Annual Decline Rate (%)",
            value=15.0,
            min_value=0.0,
            max_value=100.0,
            step=1.0
        )

    # Calculate ROI metrics
    if initial_investment > 0:
        # Generate cash flows
        months = 120  # 10 years
        cash_flows = []
        cumulative_cash_flow = -initial_investment
        monthly_production = lease_params['initial_rate']  # Initial monthly production

        for month in range(months):
            # Calculate production with decline
            production = monthly_production * np.exp(-(decline_rate / 100) * month / 12)
            revenue = production * oil_price
            costs = production * operating_cost
            net_cash_flow = revenue - costs
            cumulative_cash_flow += net_cash_flow

            cash_flows.append({
                'month': month,
                'production': production,
                'revenue': revenue,
                'costs': costs,
                'net_cash_flow': net_cash_flow,
                'cumulative_cash_flow': cumulative_cash_flow
            })

        # Convert to DataFrame for plotting
        df = pd.DataFrame(cash_flows)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Find payback period
            payback_month = df[df['cumulative_cash_flow'] >= 0].index.min()
            if pd.isna(payback_month):
                st.metric("Payback Period", "Not achieved")
            else:
                st.metric("Payback Period", f"{payback_month} months")

        with col2:
            roi = ((df['cumulative_cash_flow'].iloc[-1] + initial_investment) /
                   initial_investment * 100)
            st.metric("Total ROI", f"{roi:.1f}%")

        with col3:
            npv = calculate_npv(df['net_cash_flow'].values, 0.10, initial_investment)
            st.metric("NPV (10%)", f"${npv / 1e6:.1f}M")

        with col4:
            monthly_avg = df['net_cash_flow'].mean()
            st.metric("Avg Monthly Cash Flow", f"${monthly_avg:,.0f}")

        # Plot cash flows
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df['cumulative_cash_flow'],
            name='Cumulative Cash Flow',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df['net_cash_flow'],
            name='Monthly Cash Flow',
            line=dict(color='green')
        ))

        fig.update_layout(
            title="Cash Flow Analysis",
            xaxis_title="Month",
            yaxis_title="Cash Flow ($)",
            hovermode='x unified'
        )

        st.plotly_chart(fig)

        # Show detailed metrics table
        if st.checkbox("Show Detailed Metrics"):
            st.dataframe(
                df.style.format({
                    'production': '{:.0f}',
                    'revenue': '${:,.0f}',
                    'costs': '${:,.0f}',
                    'net_cash_flow': '${:,.0f}',
                    'cumulative_cash_flow': '${:,.0f}'
                })
            )

def main():
    """Main entry point for the Streamlit application."""
    # Set page config - must be the first Streamlit command
    st.set_page_config(
        page_title="Well Analysis Dashboard",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .reportview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
        }
        div[data-testid="stMetricValue"] {
            font-size: 20px;
        }
        .small-font {
            font-size: 14px;
        }
        div[data-testid="stSidebarNav"] {
            background-image: none;
            padding-top: 0;
        }
        div[data-testid="stSidebarNav"]::before {
            content: "Parameters";
            margin-left: 20px;
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Main title
    st.title("🛢️ Well Analysis Dashboard")
    st.markdown("---")

    # Force refresh button in sidebar
    if st.sidebar.button("Reset All Parameters", key="reset_all"):
        initialized = st.session_state.get('initialized', True)
        current_page = st.session_state.get('current_page', "Production Analysis")
        st.session_state.clear()
        st.session_state.initialized = initialized
        st.session_state.current_page = current_page
        initialize_session_state()
        st.experimental_rerun()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Production Analysis",
        "Economic Analysis",
        "Monte Carlo Simulation",
        "Lease Analysis",
        "Abandonment Analysis",
        "Technical Analysis",
        "Regulatory Compliance"
    ])

    # Run analyses in their respective tabs
    with tab1:
        st.session_state.current_page = "Production Analysis"
        create_page_sidebar(tab_name="Production Analysis", prefix="prod1")
        try:
            run_production_analysis()
        except Exception as e:
            st.error(f"Error in Production Analysis: {str(e)}")
            st.exception(e)

    with tab2:
        st.session_state.current_page = "Economic Analysis"
        create_page_sidebar(tab_name="Economic Analysis", prefix="econ2")
        try:
            run_economic_analysis()
        except Exception as e:
            st.error(f"Error in Economic Analysis: {str(e)}")
            st.exception(e)

    with tab3:
        st.session_state.current_page = "Monte Carlo Simulation"
        create_page_sidebar(tab_name="Monte Carlo Simulation", prefix="mc3")
        try:
            run_monte_carlo_analysis()
        except Exception as e:
            st.error(f"Error in Monte Carlo Simulation: {str(e)}")
            st.exception(e)

    with tab4:
        st.session_state.current_page = "Lease Analysis"
        create_page_sidebar(tab_name="Lease Analysis", prefix="lease4")
        try:
            run_lease_analysis()
        except Exception as e:
            st.error(f"Error in Lease Analysis: {str(e)}")
            st.exception(e)

    with tab5:
        st.session_state.current_page = "Abandonment Analysis"
        create_page_sidebar(tab_name="Abandonment Analysis", prefix="aband5")
        try:
            run_abandonment_analysis()
        except Exception as e:
            st.error(f"Error in Abandonment Analysis: {str(e)}")
            st.exception(e)

    with tab6:
        st.session_state.current_page = "Technical Analysis"
        create_page_sidebar(tab_name="Technical Analysis", prefix="tech6")
        try:
            run_technical_analysis()
        except Exception as e:
            st.error(f"Error in Technical Analysis: {str(e)}")
            st.exception(e)

    with tab7:
        st.session_state.current_page = "Regulatory Compliance"
        create_page_sidebar(tab_name="Regulatory Compliance", prefix="reg7")
        try:
            run_regulatory_analysis()
        except Exception as e:
            st.error(f"Error in Regulatory Analysis: {str(e)}")
            st.exception(e)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: gray; padding: 10px;'>"
            "Well Analysis Dashboard v1.0.0 | © 2024"
            "</div>",
            unsafe_allow_html=True
        )

    # Debug information in expander
    with st.expander("Debug Information", expanded=False):
        st.write("Current Page:", st.session_state.current_page)
        st.write("Session State Keys:", list(st.session_state.keys()))
        if st.button("Clear Session State", key="clear_state"):
            st.session_state.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred!")
        st.exception(e)
