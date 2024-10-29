# streamlit_app.py
import logging
import importlib
import sys
import types
import time
import random
import streamlit as st

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_unique_key(prefix):
    return f"{prefix}_{int(time.time() * 1000)}_{random.randint(0, 1000000)}"

if 'used_keys' not in st.session_state:
    st.session_state.used_keys = set()

def get_unique_key(prefix):
    key = generate_unique_key(prefix)
    st.session_state.used_keys.add(key)
    logger.debug(f"Generated unique key: {key}")
    return key

def remove_all_widget_keys():
    keys_to_remove = [key for key in list(st.session_state.keys()) if not key.startswith('_') and key != 'used_keys']
    for key in keys_to_remove:
        del st.session_state[key]
    st.session_state.used_keys = set()
    logger.debug(f"Removed all widget keys: {keys_to_remove}")
    return keys_to_remove

def clear_session_state():
    st.session_state.clear()
    st.session_state.used_keys = set()
    logger.debug("Cleared session state and reset used_keys")


def remove_old_format_keys():
    pass


def clear_all_widgets():
    logger.debug("Clearing all widgets and cached keys")
    for key in list(st.session_state.keys()):
        if not key.startswith('_'):
            del st.session_state[key]
    st.session_state.clear()
    st.session_state.used_keys = set()
    remove_old_format_keys()
    logger.debug("All widgets and cached keys cleared")

def clear_cache_and_rerun():
    logger.debug("Clearing cache and forcing rerun")
    st.cache_data.clear()
    st.cache_resource.clear()
    clear_all_widgets()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

def log_all_keys():
    logger.debug(f"All keys in session state: {list(st.session_state.keys())}")
    logger.debug(f"Used keys: {st.session_state.used_keys}")

def force_refresh():
    st.session_state.clear()
    clear_cache_and_rerun()

def debug_module_state(module_name):
    logger.debug(f"Debugging {module_name}")
    if module_name in sys.modules:
        module = sys.modules[module_name]
        logger.debug(f"{module_name} is in sys.modules")
        logger.debug(f"{module_name}.__file__: {getattr(module, '__file__', 'N/A')}")
        logger.debug(f"{module_name}.__dict__ keys: {list(module.__dict__.keys())}")
        if 'EconomicParameters' in module.__dict__:
            logger.debug(f"EconomicParameters in {module_name}: {module.EconomicParameters}")
            logger.debug(f"EconomicParameters.__annotations__: {module.EconomicParameters.__annotations__}")
    else:
        logger.debug(f"{module_name} is not in sys.modules")

logger.debug("Initial state")
debug_module_state('src.utils.economics')
debug_module_state('src.utils.economic_parameters')

logger.debug("Attempting to reload modules")
import src.utils
importlib.reload(src.utils)
import src.utils.economic_parameters
import src.utils.economics
importlib.reload(src.utils.economic_parameters)
importlib.reload(src.utils.economics)
from src.utils.economic_parameters import EconomicParameters
from src.utils.economics import EconomicAnalyzer

logger.debug("After reloading")
debug_module_state('src.utils.economics')
debug_module_state('src.utils.economic_parameters')

logger.debug(f"EconomicParameters at global scope: {EconomicParameters}")
logger.debug(f"EconomicParameters fields at global scope: {EconomicParameters.__annotations__}")

# Streamlit cache clear
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()

logger.debug("Final state")
debug_module_state('src.utils.economics')
debug_module_state('src.utils.economic_parameters')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path

# Import custom modules
from src.utils.monte_carlo import MonteCarloSimulator
import importlib
import src.utils.monte_carlo
importlib.reload(src.utils.monte_carlo)
from src.utils.monte_carlo import MonteCarloSimulator, MonteCarloResults
from src.utils.economic_parameters import EconomicParameters
from src.utils.economics import EconomicAnalyzer
from src.utils.equipment import EquipmentAnalyzer
from src.utils.lease_analysis import LeaseAnalyzer, LeaseTerms, AbandonmentCosts

# Import all necessary functions from src.utils
from src.utils import (
    create_production_plot,
    create_costs_plot,
    create_cash_flow_plot,
    create_monte_carlo_plots,
    create_abandonment_plots,
    create_technical_plots,
    create_tornado_plot,
    create_equipment_health_plot
)

# Configuration and Setup
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
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    # Force reload of all relevant modules
    import importlib
    import sys
    import inspect

    # Remove monte_carlo module from sys.modules to force a complete reload
    if 'src.utils.monte_carlo' in sys.modules:
        del sys.modules['src.utils.monte_carlo']

    import src.utils.monte_carlo
    importlib.reload(src.utils.monte_carlo)
    from src.utils.monte_carlo import MonteCarloSimulator

    # Log the source code of the MonteCarloSimulator class in the reloaded module
    logger.debug(f"Reloaded MonteCarloSimulator class:\n{inspect.getsource(MonteCarloSimulator)}")

    # Always reinitialize the MonteCarloSimulator
    st.session_state.monte_carlo = MonteCarloSimulator()

    # Reinitialize other components if not present
    if 'economic' not in st.session_state:
        st.session_state.economic = EconomicAnalyzer()
    if 'equipment' not in st.session_state:
        st.session_state.equipment = EquipmentAnalyzer()
    if 'lease' not in st.session_state:
        st.session_state.lease = LeaseAnalyzer()
    if 'current_data' not in st.session_state:
        st.session_state.current_data = create_initial_data()

    # Log the current state of MonteCarloSimulator
    logger.debug(f"MonteCarloSimulator methods: {dir(st.session_state.monte_carlo)}")
    logger.debug(f"run_full_analysis signature: {inspect.signature(st.session_state.monte_carlo.run_full_analysis)}")

    # Force update of MonteCarloSimulator class and method
    st.session_state.monte_carlo.__class__ = MonteCarloSimulator
    st.session_state.monte_carlo.run_full_analysis = MonteCarloSimulator.run_full_analysis.__get__(st.session_state.monte_carlo, MonteCarloSimulator)

    # Verify the presence of working_interest in run_full_analysis
    run_full_analysis_params = inspect.signature(st.session_state.monte_carlo.run_full_analysis).parameters
    if 'working_interest' in run_full_analysis_params:
        logger.debug("working_interest parameter is present in run_full_analysis")
    else:
        logger.warning("working_interest parameter is NOT present in run_full_analysis")
        logger.debug(f"Available parameters: {list(run_full_analysis_params.keys())}")

    logger.debug(f"Final run_full_analysis signature: {inspect.signature(st.session_state.monte_carlo.run_full_analysis)}")

    # Add a version check to ensure we're using the latest version
    if not hasattr(st.session_state, 'monte_carlo_version'):
        st.session_state.monte_carlo_version = 0
    st.session_state.monte_carlo_version += 1
    logger.debug(f"MonteCarloSimulator version: {st.session_state.monte_carlo_version}")

    # Log the source code of the run_full_analysis method
    logger.debug(f"run_full_analysis source code:\n{inspect.getsource(st.session_state.monte_carlo.run_full_analysis)}")

    # Compare the source code of the run_full_analysis method in the reloaded module with the one in the instance
    module_method = MonteCarloSimulator.run_full_analysis
    instance_method = st.session_state.monte_carlo.run_full_analysis
    if inspect.getsource(module_method) == inspect.getsource(instance_method):
        logger.debug("run_full_analysis method in the instance matches the one in the reloaded module")
    else:
        logger.warning("run_full_analysis method in the instance does NOT match the one in the reloaded module")
        logger.debug(f"Module method:\n{inspect.getsource(module_method)}")
        logger.debug(f"Instance method:\n{inspect.getsource(instance_method)}")

    # Ensure the run_full_analysis method is bound to the instance
    st.session_state.monte_carlo.run_full_analysis = types.MethodType(MonteCarloSimulator.run_full_analysis, st.session_state.monte_carlo)

    logger.debug(f"Final run_full_analysis method: {st.session_state.monte_carlo.run_full_analysis}")
    logger.debug(f"Final run_full_analysis signature: {inspect.signature(st.session_state.monte_carlo.run_full_analysis)}")


def create_initial_data():
    """Create initial dataset structure"""
    return {
        'production': pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=30),
            'oil_rate': np.random.normal(1000, 50, 30),
            'water_cut': np.random.uniform(0.2, 0.4, 30),
            'gas_rate': np.random.normal(2000, 100, 30),
            'pressure': np.random.normal(2000, 50, 30)
        }),
        'equipment': {
            'pump_vibration': 0.3,
            'motor_temperature': 165,
            'separator_pressure': 250,
            'last_maintenance': datetime.now() - timedelta(days=45)
        },
        'technical': {
            'reservoir_pressure': 3000,
            'formation_volume_factor': 1.2,
            'temperature': 180,
            'api_gravity': 35,
            'water_saturation': 30,
            'porosity': 20
        }
    }


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}


def calculate_technical_parameters(
        pressure: float,
        temperature: float,
        api_gravity: float
) -> Dict[str, float]:
    """Calculate technical parameters for the well"""
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

    return {
        'solution_gas': rsb,
        'formation_volume_factor': bob,
        'oil_viscosity': oil_visc
    }


def main():
    logger.debug("Entering main()")

    # Force refresh button
    if st.button("Force Refresh"):
        clear_session_state()
        st.experimental_rerun()

    # Remove all widget keys at the start of each run
    removed_keys = remove_all_widget_keys()
    logger.debug(f"Removed all widget keys at start: {removed_keys}")

    # Initialize session state
    initialize_session_state()

    # Load configurations
    config = load_config('src/config/settings.yaml')

    # Main title
    st.title("🛢️ Complete Well Analysis Dashboard")

    # Log all keys before creating widgets
    log_all_keys()

    # Create sidebar inputs
    create_sidebar_inputs()

    # Log all keys after creating widgets
    log_all_keys()

    # Check for any duplicate keys
    all_keys = list(st.session_state.keys())
    if len(all_keys) != len(set(all_keys)):
        logger.error(f"Duplicate keys found in session state: {all_keys}")
        st.error("Duplicate keys detected. Please click the 'Force Refresh' button to resolve the issue.")
        return  # Exit the main function to prevent further processing

    # Update session state with the latest parameter values
    logger.debug("Updating session state parameters")
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {}

    # Only update parameters if they don't exist in session state
    if 'well' not in st.session_state.parameters:
        logger.debug("Calling get_well_params()")
        st.session_state.parameters['well'] = get_well_params()
    else:
        logger.debug("Using existing well parameters")

    if 'production' not in st.session_state.parameters:
        st.session_state.parameters['production'] = get_production_params()
    if 'economic' not in st.session_state.parameters:
        st.session_state.parameters['economic'] = get_economic_params()
    if 'technical' not in st.session_state.parameters:
        st.session_state.parameters['technical'] = get_technical_params()

    logger.debug(f"Session state parameters: {st.session_state.parameters}")
    logger.debug("Exiting main()")

    # Log the updated parameters for debugging
    logger.debug(f"Updated parameters: {st.session_state.parameters}")

    # Create main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Production Analysis",
        "Economic Analysis",
        "Monte Carlo Simulation",
        "Lease Analysis",
        "Abandonment Analysis",
        "Technical Analysis",
        "Regulatory Compliance"
    ])

    # Run tab content
    with tab1:
        run_production_analysis()
    with tab2:
        run_economic_analysis()
    with tab3:
        run_monte_carlo_analysis()
    with tab4:
        run_lease_analysis()
    with tab5:
        run_abandonment_analysis()
    with tab6:
        run_technical_analysis()
    with tab7:
        run_regulatory_analysis()


def create_sidebar_inputs():
    """Create all sidebar inputs"""
    st.sidebar.header("Input Parameters")

    # Well Information
    st.sidebar.subheader("Well Information")
    well_params = get_well_params()

    # Production Parameters
    st.sidebar.subheader("Production")
    production_params = get_production_params()

    # Economic Parameters
    st.sidebar.subheader("Economics")
    economic_params = get_economic_params()

    # Technical Parameters
    st.sidebar.subheader("Technical")
    technical_params = get_technical_params()

    # Store all parameters in session state
    st.session_state.parameters = {
        'well': well_params,
        'production': production_params,
        'economic': economic_params,
        'technical': technical_params
    }


def get_well_params() -> dict:
    """Get well parameters from sidebar"""
    logger.debug("Entering get_well_params()")

    removed_keys = remove_old_format_keys()
    logger.debug(f"Removed old format keys before creating well parameters: {removed_keys}")
    
    params = {}
    for param, (label, value, min_value) in {
        'depth': ("Well Depth (ft)", 5000.0, 0.0),
        'age': ("Well Age (years)", 5.0, 0.0),
        'type': ("Well Type", ['Oil', 'Gas', 'Dual'], None),
        'location': ("Location", "Default Field", None)
    }.items():
        key = get_unique_key(f"well_{param}")
        logger.debug(f"Processing widget with key: {key}")
        if param == 'type':
            params[param] = st.sidebar.selectbox(label, value, key=key)
        elif param == 'location':
            params[param] = st.sidebar.text_input(label, value, key=key)
        else:
            params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, key=key)
        logger.debug(f"Created widget with key: {key}, value: {params[param]}")
    
    logger.debug(f"Created well parameters with keys: {list(params.keys())}")
    logger.debug("Exiting get_well_params()")
    return params


def get_production_params() -> dict:
    """Get production parameters from sidebar"""
    logger.debug("Entering get_production_params()")
    params = {}
    for param, (label, value, min_value, max_value) in {
        'initial_rate': ("Initial Rate (bbl/day)", 1000.0, 0.0, None),
        'decline_rate': ("Annual Decline Rate (%)", 15.0, 0.0, 100.0),
        'forecast_months': ("Forecast Months", 120, 12, 240),
        'water_cut': ("Current Water Cut (%)", 20.0, 0.0, 100.0)
    }.items():
        key = get_unique_key(f"production_{param}")
        logger.debug(f"Processing production widget with key: {key}")
        if param == 'forecast_months':
            params[param] = st.sidebar.slider(label, min_value, max_value, value, key=key)
        else:
            params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
        if param in ['decline_rate', 'water_cut']:
            params[param] /= 100
    logger.debug(f"Created production parameters with keys: {list(params.keys())}")
    logger.debug("Exiting get_production_params()")
    return params


# Continuation of streamlit_app.py

def get_economic_params() -> dict:
    """Get economic parameters from sidebar"""
    logger.debug("Entering get_economic_params()")
    params = {}
    for param, (label, value, min_value, max_value) in {
        'oil_price': ("Oil Price ($/bbl)", 70.0, 0.0, None),
        'opex': ("Operating Cost ($/bbl)", 20.0, 0.0, None),
        'initial_investment': ("Initial Investment ($)", 1000000.0, 0.0, None),
        'discount_rate': ("Discount Rate (%)", 10.0, 0.0, 100.0),
        'working_interest': ("Working Interest (%)", 75, 0, 100),
        'net_revenue_interest': ("Net Revenue Interest (%)", 65, 0, 100),
        'lease_terms': ("Lease Terms (years)", 5, 1, 50),
        'abandonment_costs': ("Abandonment Costs ($)", 100000.0, 0.0, None)
    }.items():
        key = get_unique_key(f"economic_{param}")
        logger.debug(f"Processing economic widget with key: {key}")
        if param in ['working_interest', 'net_revenue_interest']:
            params[param] = st.sidebar.slider(label, min_value, max_value, value, key=key) / 100
        else:
            params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
        if param == 'discount_rate':
            params[param] /= 100
    logger.debug(f"Economic parameters: {params}")
    logger.debug("Exiting get_economic_params()")
    return params


def get_technical_params() -> dict:
    """Get technical parameters from sidebar"""
    logger.debug("Entering get_technical_params()")
    params = {}
    for param, (label, value, min_value, max_value) in {
        'reservoir_pressure': ("Reservoir Pressure (psi)", 3000.0, 0.0, None),
        'temperature': ("Temperature (°F)", 180.0, 0.0, None),
        'api_gravity': ("API Gravity", 35.0, 0.0, None),
        'porosity': ("Porosity (%)", 20.0, 0.0, 100.0),
        'water_saturation': ("Water Saturation (%)", 30.0, 0.0, 100.0)
    }.items():
        key = get_unique_key(f"technical_{param}")
        logger.debug(f"Processing technical widget with key: {key}")
        params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
    logger.debug(f"Technical parameters: {params}")
    logger.debug("Exiting get_technical_params()")
    return params


def run_production_analysis():
    """Run production analysis tab content"""
    st.header("Production Analysis")

    params = st.session_state.parameters

    # Calculate production profile
    time = np.arange(params['production']['forecast_months'])
    production = params['production']['initial_rate'] * \
                 np.exp(-params['production']['decline_rate'] * time)

    # Create production dataframe
    prod_df = pd.DataFrame({
        'Month': time,
        'Production': production,
        'Water_Cut': np.minimum(params['production']['water_cut'] *
                                np.exp(0.01 * time), 0.95)
    })

    # Production plot
    fig = create_production_plot(prod_df)
    st.plotly_chart(fig, use_container_width=True)

    # Production metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cumulative Production", f"{production.sum():,.0f} bbl")
        st.metric("Average Rate", f"{production.mean():,.0f} bbl/d")
    with col2:
        st.metric("Initial Rate", f"{production[0]:,.0f} bbl/d")
        st.metric("Final Rate", f"{production[-1]:,.0f} bbl/d")
    with col3:
        st.metric("Decline Rate", f"{params['production']['decline_rate'] * 100:.1f}%/year")
        st.metric("Production Life", f"{len(time) / 12:.1f} years")

    # Store production data in session state
    st.session_state.production_data = prod_df


def run_economic_analysis():
    """Run economic analysis tab content"""
    st.header("Economic Analysis")
    
    logger.debug("Beginning of run_economic_analysis")
    
    # Force reload of the economic_parameters module
    import importlib
    import src.utils.economic_parameters
    importlib.reload(src.utils.economic_parameters)
    from src.utils.economic_parameters import EconomicParameters
    
    logger.debug(f"Reloaded EconomicParameters v{EconomicParameters.VERSION}")
    logger.debug(f"EconomicParameters at start of run_economic_analysis: {EconomicParameters}")
    logger.debug(f"EconomicParameters fields at start of run_economic_analysis: {EconomicParameters.__annotations__}")

    params = st.session_state.parameters['economic']
    prod_data = st.session_state.production_data

    logger.debug(f"params: {params}")

    # Calculate economics
    try:
        logger.debug("Attempting to create EconomicParameters instance")
        logger.debug(f"EconomicParameters class just before instantiation: {EconomicParameters}")
        logger.debug(f"EconomicParameters fields just before instantiation: {EconomicParameters.__dict__}")
        
        # Version check
        if EconomicParameters.VERSION != "1.2":
            raise ValueError(f"Incorrect EconomicParameters version. Expected 1.2, got {EconomicParameters.VERSION}")
        
        economic_params = EconomicParameters.from_dict(params)
        logger.debug(f"EconomicParameters instance created successfully: {economic_params}")
        logger.debug(f"EconomicParameters instance fields: {economic_params.__dict__}")
    except Exception as e:
        logger.error(f"Error creating EconomicParameters instance: {str(e)}")
        logger.error(f"EconomicParameters class: {EconomicParameters}")
        logger.error(f"EconomicParameters fields: {EconomicParameters.__dict__}")
        logger.error(f"Params being passed: {params}")
        st.error(f"An error occurred while creating EconomicParameters: {str(e)}")
        return

    results = st.session_state.economic.calculate_metrics(
        production=prod_data['Production'].values,
        params=economic_params
    )

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Net Present Value", f"${results.npv:,.0f}")
        st.metric("ROI", f"{results.roi:.1f}%")
    with col2:
        st.metric("Payback Period",
                  f"{results.payback_period:.1f} months"
                  if results.payback_period < len(prod_data)
                  else "Not achieved")
        st.metric("Net Profit", f"${results.net_profit:,.0f}")
    with col3:
        st.metric("Total Revenue", f"${results.total_revenue:,.0f}")
        st.metric("Total Costs", f"${results.total_costs:,.0f}")

    # Cash flow analysis
    st.subheader("Cash Flow Analysis")
    cash_flow_df = pd.DataFrame({
        'Month': prod_data['Month'],
        'Net_Revenue': results.monthly_cash_flow,
        'Cumulative_Cash_Flow': np.cumsum(results.monthly_cash_flow)
    })

    fig = create_cash_flow_plot(cash_flow_df)
    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown
    st.subheader("Cost Analysis")
    costs_df = pd.DataFrame({
        'Month': prod_data['Month'],
        'Operating': prod_data['Production'] * params['opex'],
        'Water_Handling': prod_data['Production'] * prod_data['Water_Cut'] * params['opex'] * 0.3,
        'Maintenance': prod_data['Production'] * params['opex'] * 0.1
    })

    fig = create_costs_plot(costs_df, ['Operating', 'Water_Handling', 'Maintenance'])
    st.plotly_chart(fig, use_container_width=True)


def run_monte_carlo_analysis():
    """Run Monte Carlo analysis tab content"""
    st.header("Monte Carlo Simulation")

    params = st.session_state.parameters

    # Log the raw parameters for debugging
    logger.debug(f"Raw parameters: {params}")

    # Combine economic and production parameters
    combined_params = {
        **params['economic'],
        'initial_rate': params['production']['initial_rate'],
        'decline_rate': params['production']['decline_rate']
    }

    # Ensure all required parameters are in the combined_params
    required_params = ['oil_price', 'opex', 'initial_investment', 'discount_rate', 'initial_rate', 'decline_rate',
                       'working_interest', 'net_revenue_interest', 'lease_terms', 'abandonment_costs']
    for param in required_params:
        if param not in combined_params:
            if param in params['economic']:
                combined_params[param] = params['economic'][param]
            elif param in params['production']:
                combined_params[param] = params['production'][param]
            else:
                logger.error(f"Missing required parameter: {param}")
                st.error(f"An error occurred: Missing required parameter: {param}")
                return

    # Log the combined parameters for debugging
    logger.debug(f"Combined parameters for EconomicParameters: {combined_params}")
    logger.debug(f"Lease terms in combined parameters: {combined_params.get('lease_terms', 'Not found')}")

    # Create EconomicParameters instance
    try:
        economic_params = EconomicParameters(**combined_params)
    except Exception as e:
        logger.error(f"Error creating EconomicParameters: {str(e)}")
        st.error(f"An error occurred while creating EconomicParameters: {str(e)}")
        return

    # Log the created EconomicParameters instance for debugging
    logger.debug(f"Created EconomicParameters instance: {economic_params.__dict__}")

    # Log the types of parameters
    logger.debug(f"Parameter types: {[(k, type(v)) for k, v in combined_params.items()]}")

    # Verify lease_terms is set correctly
    logger.debug(f"Lease terms in EconomicParameters: {getattr(economic_params, 'lease_terms', 'Not found')}")

    # Run simulation
    try:
        results = st.session_state.monte_carlo.run_full_analysis(
            economic_params=economic_params,
            months=params['production']['forecast_months'],
            iterations=1000,
            confidence_level=0.90
        )
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {str(e)}")
        st.error(f"An error occurred while running the Monte Carlo simulation: {str(e)}")
        return

    # Log the simulation results for debugging
    logger.debug(f"Monte Carlo simulation results: {results}")

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
                f"{results.percentiles['NPV']['P10']:,.0f}",
                f"{results.percentiles['NPV']['P50']:,.0f}",
                f"{results.percentiles['NPV']['P90']:,.0f}",
                f"{results.statistics['NPV']['mean']:,.0f}",
                f"{results.statistics['NPV']['std']:,.0f}"
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


def run_lease_analysis():
    """Run lease analysis tab content"""
    st.header("Lease Analysis")

    params = st.session_state.parameters

    # Calculate lease metrics
    lease_terms = LeaseTerms(
        working_interest=params['economic']['working_interest'],
        net_revenue_interest=params['economic']['net_revenue_interest'],
        royalty_rate=0.20,  # Standard royalty rate
        lease_bonus=50000,
        lease_term_years=5,
        extension_cost=25000,
        minimum_royalty=1000
    )

    lease_results = st.session_state.lease.calculate_lease_economics(
        production=st.session_state.production_data['Production'].values,
        oil_price=params['economic']['oil_price'],
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


# Continuation of streamlit_app.py

def run_abandonment_analysis():
    """Run abandonment analysis tab content"""
    st.header("Abandonment Analysis")

    params = st.session_state.parameters

    # Calculate abandonment costs
    abandonment_costs = AbandonmentCosts(
        plugging_cost=params['well']['depth'] * 25,  # $25 per foot
        site_restoration=params['well']['depth'] * 5,  # $5 per foot
        equipment_removal=50000,  # Base equipment removal cost
        environmental_cleanup=75000,  # Base environmental cleanup cost
        regulatory_fees=5000,  # Base regulatory fees
        contingency=0.15  # 15% contingency
    )

    # Calculate total obligation
    obligation = st.session_state.lease.calculate_total_abandonment_obligation(
        costs=abandonment_costs,
        inflation_years=params['well']['age']
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
            f"${obligation['total_obligation'] / params['well']['depth']:,.2f}/ft"
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
    economic_limit = params['economic']['opex'] * 1.5  # 1.5x operating cost
    production = st.session_state.production_data['Production'].values
    revenue = production * params['economic']['oil_price']

    months_to_limit = np.where(revenue < economic_limit * production)[0]
    if len(months_to_limit) > 0:
        economic_limit_date = datetime.now() + timedelta(days=30 * months_to_limit[0])
        st.warning(f"⚠️ Economic limit reached in {months_to_limit[0]} months "
                   f"(approximately {economic_limit_date.strftime('%B %Y')})")
    else:
        st.success("✅ Well remains economic throughout forecast period")


def run_technical_analysis():
    """Run technical analysis tab content"""
    st.header("Technical Analysis")

    params = st.session_state.parameters['technical']

    # Calculate technical parameters
    tech_params = calculate_technical_parameters(
        pressure=params['reservoir_pressure'],
        temperature=params['temperature'],
        api_gravity=params['api_gravity']
    )

    # Display reservoir parameters
    st.subheader("Reservoir Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Reservoir Pressure", f"{params['reservoir_pressure']:,.0f} psi")
        st.metric("Formation Volume Factor", f"{tech_params['formation_volume_factor']:.3f} RB/STB")

    with col2:
        st.metric("Temperature", f"{params['temperature']}°F")
        st.metric("Oil Viscosity", f"{tech_params['oil_viscosity']:.2f} cp")

    with col3:
        st.metric("API Gravity", f"{params['api_gravity']:.1f}°API")
        st.metric("Solution GOR", f"{tech_params['solution_gas']:,.0f} scf/STB")

    # PVT Analysis
    st.subheader("PVT Analysis")
    pressures = np.linspace(0, params['reservoir_pressure'], 50)
    pvt_data = pd.DataFrame({
        'Pressure': pressures,
        'Oil_FVF': [tech_params['formation_volume_factor'] * (1 - 0.1 * p / params['reservoir_pressure'])
                    for p in pressures],
        'Viscosity': [tech_params['oil_viscosity'] * (1 + 0.1 * p / params['reservoir_pressure'])
                      for p in pressures]
    })

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Oil FVF vs Pressure", "Oil Viscosity vs Pressure"))
    fig.add_trace(go.Scatter(x=pvt_data['Pressure'], y=pvt_data['Oil_FVF'], name='Oil FVF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pvt_data['Pressure'], y=pvt_data['Viscosity'], name='Oil Viscosity'), row=1, col=2)
    fig.update_layout(height=400, title_text="PVT Analysis", showlegend=True)
    fig.update_xaxes(title_text="Pressure (psi)")
    fig.update_yaxes(title_text="Oil FVF (RB/STB)", row=1, col=1)
    fig.update_yaxes(title_text="Oil Viscosity (cp)", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # Nodal Analysis
    st.subheader("Nodal Analysis")

    # Calculate IPR and TPR
    q_max = st.session_state.parameters['production']['initial_rate'] * 1.5
    ipr_data = pd.DataFrame({
        'Rate': np.linspace(0, q_max, 50),
        'Pressure': [params['reservoir_pressure'] * (1 - (q / q_max) ** 2)
                     for q in np.linspace(0, q_max, 50)]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ipr_data['Rate'], y=ipr_data['Pressure'], name='IPR'))
    fig.update_layout(title="Inflow Performance Relationship (IPR)", xaxis_title="Flow Rate (STB/d)", yaxis_title="Pressure (psi)")
    st.plotly_chart(fig, use_container_width=True)


def run_regulatory_analysis():
    """Run regulatory analysis tab content"""
    st.header("Regulatory Compliance")

    # Compliance Status
    st.subheader("Compliance Status")

    # Environmental Compliance
    env_compliance = {
        'Air Emissions': {'status': 'Compliant', 'last_inspection': '2024-01-15',
                          'next_due': '2024-07-15'},
        'Water Disposal': {'status': 'Warning', 'last_inspection': '2023-12-01',
                           'next_due': '2024-06-01'},
        'Spill Prevention': {'status': 'Compliant', 'last_inspection': '2024-02-01',
                             'next_due': '2024-08-01'},
        'Site Maintenance': {'status': 'Action Required', 'last_inspection': '2023-11-15',
                             'next_due': '2024-05-15'}
    }

    for category, details in env_compliance.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if details['status'] == 'Compliant':
                st.success(f"{category}: {details['status']}")
            elif details['status'] == 'Warning':
                st.warning(f"{category}: {details['status']}")
            else:
                st.error(f"{category}: {details['status']}")
        with col2:
            st.write(f"Last: {details['last_inspection']}")
        with col3:
            st.write(f"Next: {details['next_due']}")

    # Permit Status
    st.subheader("Permit Status")
    permits = pd.DataFrame({
        'Permit Type': ['Operating Permit', 'Water Disposal', 'Air Quality', 'Land Use'],
        'Status': ['Active', 'Active', 'Renewal Required', 'Active'],
        'Expiration Date': ['2025-01-01', '2024-08-15', '2024-04-01', '2024-12-31'],
        'Renewal Cost': [5000, 3000, 2500, 1500]
    })

    # Apply color coding to status
    def color_status(val):
        if val == 'Active':
            return 'background-color: #90EE90'
        elif val == 'Renewal Required':
            return 'background-color: #FFB6C1'
        return ''

    st.dataframe(permits.style.applymap(color_status, subset=['Status']))

    # Upcoming Requirements
    st.subheader("Upcoming Requirements")
    upcoming = pd.DataFrame({
        'Requirement': [
            'Quarterly Production Report',
            'Annual Pressure Test',
            'Environmental Impact Assessment',
            'Safety System Inspection'
        ],
        'Due Date': [
            '2024-04-15',
            '2024-06-30',
            '2024-08-15',
            '2024-05-01'
        ],
        'Estimated Cost': [500, 2500, 15000, 3500]
    })

    upcoming['Days Until Due'] = (pd.to_datetime(upcoming['Due Date']) -
                                  pd.Timestamp.now()).dt.days

    for _, req in upcoming.iterrows():
        days_left = req['Days Until Due']
        if days_left < 30:
            st.error(f"⚠️ {req['Requirement']} due in {days_left} days")
        elif days_left < 60:
            st.warning(f"⚡ {req['Requirement']} due in {days_left} days")
        else:
            st.info(f"📅 {req['Requirement']} due in {days_left} days")

    # Total regulatory costs
    total_permit_cost = permits['Renewal Cost'].sum()
    total_upcoming_cost = upcoming['Estimated Cost'].sum()
    total_regulatory = total_permit_cost + total_upcoming_cost

    st.metric(
        "Total Regulatory Cost",
        f"${total_regulatory:,.2f}",
        delta=f"${total_regulatory / 12:,.2f}/month"
    )


if __name__ == "__main__":
    main()