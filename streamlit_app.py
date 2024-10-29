# streamlit_app.py

import logging
from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

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
def generate_unique_key(prefix: str) -> str:
    """Generate a unique key for Streamlit widgets."""
    import time
    import random
    return f"{prefix}_{int(time.time() * 1000)}_{random.randint(0, 1000000)}"


def get_unique_key(prefix: str) -> str:
    """Get a unique key and store it in session state."""
    if 'used_keys' not in st.session_state:
        st.session_state.used_keys = set()
    key = generate_unique_key(prefix)
    st.session_state.used_keys.add(key)
    return key


def remove_all_widget_keys():
    """Remove all widget keys from session state."""
    keys_to_remove = [key for key in list(st.session_state.keys())
                      if not key.startswith('_') and key != 'used_keys']
    for key in keys_to_remove:
        del st.session_state[key]
    st.session_state.used_keys = set()
    return keys_to_remove


def clear_session_state():
    """Clear the entire session state."""
    st.session_state.clear()
    st.session_state.used_keys = set()


def clear_all_widgets():
    """Clear all widgets and cached keys."""
    for key in list(st.session_state.keys()):
        if not key.startswith('_'):
            del st.session_state[key]
    st.session_state.clear()
    st.session_state.used_keys = set()


def clear_cache_and_rerun():
    """Clear cache and force rerun."""
    st.cache_data.clear()
    st.cache_resource.clear()
    clear_all_widgets()
    st.experimental_rerun()


def load_config(config_path: str = 'src/config/settings.yaml') -> dict:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}


def create_initial_data():
    """Create initial dataset structure."""
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


def initialize_session_state():
    """Initialize session state with all required components."""
    if 'monte_carlo' not in st.session_state:
        st.session_state.monte_carlo = MonteCarloSimulator()
    if 'economic' not in st.session_state:
        st.session_state.economic = EconomicAnalyzer()
    if 'equipment' not in st.session_state:
        st.session_state.equipment = EquipmentAnalyzer()
    if 'lease' not in st.session_state:
        st.session_state.lease = LeaseAnalyzer()

    # Initialize parameters if not present
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {
            'well': {},
            'production': {},
            'economic': {},
            'technical': {}
        }

    if 'current_data' not in st.session_state:
        st.session_state.current_data = create_initial_data()


def update_session_parameters(section: str, params: dict):
    """Update session parameters while preserving existing values."""
    if 'parameters' not in st.session_state:
        st.session_state.parameters = {}
    if section not in st.session_state.parameters:
        st.session_state.parameters[section] = {}
    st.session_state.parameters[section].update(params)
# Parameter Input Functions
def get_well_params() -> dict:
    """Get well parameters from sidebar."""
    params = {}
    for param, (label, value, min_value) in {
        'depth': ("Well Depth (ft)", 5000.0, 0.0),
        'age': ("Well Age (years)", 5.0, 0.0),
        'type': ("Well Type", ['Oil', 'Gas', 'Dual'], None),
        'location': ("Location", "Default Field", None)
    }.items():
        key = get_unique_key(f"well_{param}")
        if param == 'type':
            params[param] = st.sidebar.selectbox(label, value, key=key)
        elif param == 'location':
            params[param] = st.sidebar.text_input(label, value, key=key)
        else:
            params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, key=key)
    return params

def get_production_params() -> dict:
    """Get production parameters from sidebar."""
    params = {}
    for param, (label, value, min_value, max_value) in {
        'initial_rate': ("Initial Rate (bbl/day)", 1000.0, 0.0, None),
        'decline_rate': ("Annual Decline Rate (%)", 15.0, 0.0, 100.0),
        'forecast_months': ("Forecast Months", 120, 12, 240),
        'water_cut': ("Current Water Cut (%)", 20.0, 0.0, 100.0)
    }.items():
        key = get_unique_key(f"production_{param}")
        if param == 'forecast_months':
            params[param] = st.sidebar.slider(label, min_value, max_value, value, key=key)
        else:
            params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
        if param in ['decline_rate', 'water_cut']:
            params[param] /= 100  # Convert percentage to decimal
    return params

def get_economic_params() -> dict:
    """Get economic parameters from sidebar."""
    params = {}
    for param, (label, value, min_value, max_value) in {
        'oil_price': ("Oil Price ($/bbl)", 70.0, 0.0, None),
        'opex': ("Operating Cost ($/bbl)", 20.0, 0.0, None),
        'initial_investment': ("Initial Investment ($)", 1000000.0, 0.0, None),
        'discount_rate': ("Discount Rate (%)", 10.0, 0.0, 100.0),
        'working_interest': ("Working Interest (%)", 75.0, 0.0, 100.0),
        'net_revenue_interest': ("Net Revenue Interest (%)", 65.0, 0.0, 100.0),
        'lease_terms': ("Lease Terms (years)", 5.0, 1.0, 50.0),
        'abandonment_costs': ("Abandonment Costs ($)", 100000.0, 0.0, None)
    }.items():
        key = get_unique_key(f"economic_{param}")
        params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
        if param in ['discount_rate', 'working_interest', 'net_revenue_interest']:
            params[param] /= 100  # Convert percentage to decimal
    return params

def get_technical_params() -> dict:
    """Get technical parameters from sidebar."""
    params = {}
    for param, (label, value, min_value, max_value) in {
        'reservoir_pressure': ("Reservoir Pressure (psi)", 3000.0, 0.0, None),
        'temperature': ("Temperature (°F)", 180.0, 0.0, None),
        'api_gravity': ("API Gravity", 35.0, 0.0, None),
        'porosity': ("Porosity (%)", 20.0, 0.0, 100.0),
        'water_saturation': ("Water Saturation (%)", 30.0, 0.0, 100.0)
    }.items():
        key = get_unique_key(f"technical_{param}")
        params[param] = st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
        if param in ['porosity', 'water_saturation']:
            params[param] /= 100  # Convert percentage to decimal
    return params

def create_sidebar_inputs():
    """Create all sidebar inputs."""
    st.sidebar.header("Input Parameters")

    # Well Information
    st.sidebar.subheader("Well Information")
    well_params = get_well_params()
    update_session_parameters('well', well_params)

    # Production Parameters
    st.sidebar.subheader("Production")
    production_params = get_production_params()
    update_session_parameters('production', production_params)

    # Economic Parameters
    st.sidebar.subheader("Economics")
    economic_params = get_economic_params()
    update_session_parameters('economic', economic_params)

    # Technical Parameters
    st.sidebar.subheader("Technical")
    technical_params = get_technical_params()
    update_session_parameters('technical', technical_params)

def calculate_technical_parameters(
    pressure: float,
    temperature: float,
    api_gravity: float
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

    return {
        'solution_gas': rsb,
        'formation_volume_factor': bob,
        'oil_viscosity': oil_visc
    }


# Analysis Functions
def run_production_analysis():
    """Run production analysis tab content."""
    st.header("Production Analysis")

    params = st.session_state.parameters

    # Calculate production profile
    time = np.arange(params['production']['forecast_months'])
    production = params['production']['initial_rate'] * \
                 np.exp(-params['production']['decline_rate'] * time / 12)

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

    st.session_state.production_data = prod_df


def run_economic_analysis():
    """Run economic analysis tab content."""
    st.header("Economic Analysis")

    params = st.session_state.parameters['economic']
    prod_data = st.session_state.production_data

    # Create EconomicParameters instance
    try:
        economic_params = EconomicParameters(
            oil_price=params['oil_price'],
            opex=params['opex'],
            initial_investment=params['initial_investment'],
            discount_rate=params['discount_rate'],
            initial_rate=st.session_state.parameters['production']['initial_rate'],
            decline_rate=st.session_state.parameters['production']['decline_rate'],
            working_interest=params['working_interest'],
            net_revenue_interest=params['net_revenue_interest'],
            lease_terms=params['lease_terms'],
            abandonment_costs=params['abandonment_costs']
        )
    except Exception as e:
        st.error(f"Error creating economic parameters: {str(e)}")
        return

    # Calculate economics
    try:
        results = st.session_state.economic.calculate_metrics(
            production=prod_data['Production'].values,
            params=economic_params
        )
    except Exception as e:
        st.error(f"Error calculating economic metrics: {str(e)}")
        return

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
    """Run Monte Carlo analysis tab content."""
    st.header("Monte Carlo Simulation")

    params = st.session_state.parameters

    # Simulation settings in sidebar
    st.sidebar.subheader("Monte Carlo Settings")
    iterations = st.sidebar.number_input(
        "Number of Iterations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key=get_unique_key("mc_iterations")
    )
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.8,
        max_value=0.99,
        value=0.90,
        step=0.01,
        key=get_unique_key("mc_confidence")
    )

    # Create EconomicParameters instance
    try:
        economic_params = EconomicParameters(
            oil_price=params['economic']['oil_price'],
            opex=params['economic']['opex'],
            initial_investment=params['economic']['initial_investment'],
            discount_rate=params['economic']['discount_rate'],
            initial_rate=params['production']['initial_rate'],
            decline_rate=params['production']['decline_rate'],
            working_interest=params['economic']['working_interest'],
            net_revenue_interest=params['economic']['net_revenue_interest'],
            lease_terms=params['economic']['lease_terms'],
            abandonment_costs=params['economic']['abandonment_costs']
        )
    except Exception as e:
        st.error(f"Error creating economic parameters: {str(e)}")
        return

    # Run simulation
    try:
        with st.spinner("Running Monte Carlo simulation..."):
            results = st.session_state.monte_carlo.run_full_analysis(
                economic_params=economic_params,
                months=params['production']['forecast_months'],
                iterations=iterations,
                confidence_level=confidence_level
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
        'Oil Price': (params['economic']['oil_price'] * 0.7,
                     params['economic']['oil_price'] * 1.3),
        'OPEX': (params['economic']['opex'] * 0.8,
                 params['economic']['opex'] * 1.2),
        'Initial Rate': (params['production']['initial_rate'] * 0.9,
                        params['production']['initial_rate'] * 1.1),
        'Decline Rate': (params['production']['decline_rate'] * 0.8,
                        params['production']['decline_rate'] * 1.2)
    }
    fig_tornado = create_tornado_plot(results, results.statistics['NPV']['mean'], sensitivity_ranges)
    st.plotly_chart(fig_tornado, use_container_width=True)

def run_lease_analysis():
    """Run lease analysis tab content."""
    st.header("Lease Analysis")

    params = st.session_state.parameters

    # Calculate lease metrics
    lease_terms = LeaseTerms(
        working_interest=params['economic']['working_interest'],
        net_revenue_interest=params['economic']['net_revenue_interest'],
        royalty_rate=0.20,  # Standard royalty rate
        lease_bonus=50000,
        lease_term_years=params['economic']['lease_terms'],
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

    # Display lease status
    st.subheader("Lease Status")
    if lease_results['remaining_term_months'] <= 0:
        st.warning(f"⚠️ Lease has expired. Extension cost: ${lease_results['extension_cost']:,.2f}")
    else:
        st.success(f"✅ Lease active. {lease_results['remaining_term_months']} months remaining")

def run_abandonment_analysis():
    """Run abandonment analysis tab content."""
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
    """Run technical analysis tab content."""
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

    # Technical plots
    st.subheader("Technical Analysis")
    figs = create_technical_plots(params)
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

def run_regulatory_analysis():
    """Run regulatory analysis tab content."""
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
def main():
    """Main entry point for the Streamlit application."""
    # Set page config
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

    # Force refresh button
    if st.button("Force Refresh"):
        clear_cache_and_rerun()

    # Initialize session state
    initialize_session_state()

    # Create sidebar inputs
    create_sidebar_inputs()

    # Main title
    st.title("🛢️ Complete Well Analysis Dashboard")

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

if __name__ == "__main__":
    main()