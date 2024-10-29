import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

# Import project modules
from src.calculations.production import calculate_decline_curve
from src.calculations.costs import calculate_operating_costs, calculate_maintenance_costs
from src.calculations.npv import calculate_npv, calculate_irr, calculate_break_even_price
from src.calculations.taxes import calculate_tax_obligations, calculate_severance_tax

from src.models.equipment import EquipmentComponent
from src.models.environmental import EnvironmentalRegulation
from src.models.financial import TaxCalculator, FinancialMetrics

from src.utils.plotting import (
    create_production_plot,
    create_costs_plot,
    create_cash_flow_plot,
    create_equipment_health_plot
)
from src.utils.monte_carlo import MonteCarloSimulator
from src.utils.helpers import load_config, validate_inputs, format_currency

# Configure page
st.set_page_config(
    page_title="Oil Well Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)


def load_project_data():
    """Load data from project structure"""
    data_dir = Path("data")

    try:
        data = {
            'esp_data': pd.read_csv(data_dir / "equipment_specs/pumps/esp_catalog.csv"),
            'plunger_data': pd.read_csv(data_dir / "equipment_specs/pumps/plunger_lift_specs.csv"),
            'rod_pump_data': pd.read_csv(data_dir / "equipment_specs/pumps/rod_pump_catalog.csv"),
            'vessel_data': pd.read_csv(data_dir / "equipment_specs/separators/vessel_specs.csv"),
            'tank_data': pd.read_csv(data_dir / "equipment_specs/storage/tank_specs.csv"),
            'pressure_data': pd.read_csv(data_dir / "equipment_specs/wellhead/pressure_ratings.csv"),
            'valve_data': pd.read_csv(data_dir / "equipment_specs/wellhead/valve_specs.csv"),
            'tax_data': pd.read_csv(data_dir / "tax_tables/federal/depreciation_schedules.csv"),
            'severance_data': pd.read_csv(data_dir / "tax_tables/state/severance_rates.csv")
        }

        # Load configuration files
        configs = {
            'equipment': load_config(Path("src/config/equipment.yaml")),
            'regulations': load_config(Path("src/config/regulations.yaml")),
            'tax_rates': load_config(Path("src/config/tax_rates.yaml"))
        }

        return {**data, **configs}
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def predict_equipment_failure(vibration, temperature, health):
    """Predict equipment failure using project models"""
    equipment = EquipmentComponent(
        name="ESP",
        expected_life=60,
        replacement_cost=50000,
        failure_impact=0.8,
        maintenance_schedule=3,
        operating_conditions=health / 100
    )

    parameters = {
        'vibration': vibration,
        'temperature': temperature
    }

    reliability = equipment.calculate_reliability(12)  # 12 months forecast
    return 1 - reliability


def initialize_components(data):
    """Initialize model components"""
    components = {
        'equipment': {},
        'regulations': {},
        'tax': None
    }

    if 'equipment' in data:
        for name, specs in data['equipment'].items():
            components['equipment'][name] = EquipmentComponent(**specs)

    if 'regulations' in data:
        for name, specs in data['regulations'].items():
            components['regulations'][name] = EnvironmentalRegulation(**specs)

    if 'tax_rates' in data:
        components['tax'] = TaxCalculator(**data['tax_rates']['federal'])

    return components


def main():
    st.title("🛢️ Oil Well Analysis")

    # Load project data
    data = load_project_data()
    if data is None:
        st.error("Failed to load project data. Please check file paths and permissions.")
        return

    # Initialize components
    components = initialize_components(data)

    # Initialize Monte Carlo simulator
    monte_carlo = MonteCarloSimulator()

    # Sidebar inputs
    st.sidebar.header("Input Parameters")

    # Production parameters
    st.sidebar.subheader("Production")
    initial_rate = st.sidebar.number_input("Initial Rate (bbl/day)", value=1000.0, min_value=0.0)
    decline_rate = st.sidebar.number_input("Annual Decline Rate (%)", value=15.0, min_value=0.0,
                                           max_value=100.0) / 100 / 12
    b_factor = st.sidebar.slider("Hyperbolic Decline Factor", 0.0, 1.0, 0.0)
    forecast_months = st.sidebar.slider("Forecast Months", 12, 120, 60)

    # Economic parameters
    st.sidebar.subheader("Economics")
    oil_price = st.sidebar.number_input("Oil Price ($/bbl)", value=70.0, min_value=0.0)
    opex = st.sidebar.number_input("Operating Cost ($/bbl)", value=20.0, min_value=0.0)
    discount_rate = st.sidebar.number_input("Discount Rate (%)", value=10.0, min_value=0.0, max_value=100.0) / 100

    # Equipment parameters
    st.sidebar.subheader("Equipment")
    current_health = st.sidebar.slider("Equipment Health (%)", 0, 100, 85)
    pump_vibration = st.sidebar.slider("Pump Vibration", 0.0, 1.0, 0.2)
    motor_temperature = st.sidebar.slider("Motor Temperature (°F)", 150, 250, 180)

    # Calculate production forecast
    production = calculate_decline_curve(initial_rate, decline_rate, forecast_months, b_factor)
    time = np.arange(forecast_months)

    # Calculate economics
    revenue = production * oil_price
    costs, cost_components = calculate_operating_costs(production, np.zeros_like(production), opex)
    cash_flow = revenue - costs

    npv = calculate_npv(cash_flow, discount_rate)
    irr = calculate_irr(cash_flow, initial_investment=0)
    break_even = calculate_break_even_price(production, costs, opex)

    # Calculate equipment reliability
    failure_prob = predict_equipment_failure(pump_vibration, motor_temperature, current_health)

    # Monte Carlo simulation
    simulation_results = monte_carlo.simulate_production(
        initial_rate=initial_rate,
        decline_rate=decline_rate,
        months=forecast_months,
        iterations=1000
    )
    percentiles = monte_carlo.get_percentiles(simulation_results)

    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Production",
            f"{production[0]:.0f} bbl/d",
            f"{production[1] - production[0]:.1f} bbl/d"
        )

    with col2:
        st.metric(
            "Net Cash Flow",
            format_currency(cash_flow[0], include_cents=False),
            format_currency(cash_flow[1] - cash_flow[0], include_cents=False)
        )

    with col3:
        st.metric(
            "Equipment Health",
            f"{current_health}%",
            f"{-5}%"
        )

    with col4:
        st.metric(
            "Failure Probability",
            f"{failure_prob * 100:.1f}%",
            "Critical" if failure_prob > 0.7 else "Normal"
        )

    # Create tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["Production", "Economics", "Equipment"])

    with tab1:
        st.subheader("Production Forecast")

        # Production plot
        fig = create_production_plot(pd.DataFrame({
            'Month': time,
            'Production': production,
            'Water_Cut': np.zeros_like(production)
        }))
        st.plotly_chart(fig, use_container_width=True)

        # Production metrics
        st.subheader("Production Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cumulative Production", f"{production.sum():,.0f} bbl")
        with col2:
            st.metric("Average Rate", f"{production.mean():,.0f} bbl/d")
        with col3:
            st.metric("Final Rate", f"{production[-1]:,.0f} bbl/d")

        # Monte Carlo results
        st.subheader("Production Uncertainty")
        for p, values in percentiles.items():
            st.metric(f"{p} Production", f"{values[-1]:.0f} bbl/d")

    with tab2:
        st.subheader("Economic Analysis")

        # Financial metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NPV", format_currency(npv))
        with col2:
            st.metric("IRR", f"{irr * 100:.1f}%" if irr else "N/A")
        with col3:
            st.metric("Break-even Price", format_currency(break_even, 2))

        # Cash flow plot
        fig = create_cash_flow_plot(pd.DataFrame({
            'Month': time,
            'Net_Revenue': cash_flow
        }))
        st.plotly_chart(fig, use_container_width=True)

        # Cost breakdown
        st.subheader("Cost Analysis")
        fig = create_costs_plot(
            pd.DataFrame({
                'Month': time,
                **cost_components
            }),
            list(cost_components.keys())
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Equipment Analysis")

        # Equipment health gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Equipment Health"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ]
            }
        ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Equipment metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pump Vibration", f"{pump_vibration:.2f}")
            st.metric("Motor Temperature", f"{motor_temperature}°F")
        with col2:
            st.metric("Failure Probability", f"{failure_prob * 100:.1f}%")
            days_to_maintenance = max(0, int((0.7 - failure_prob) / 0.01))
            st.metric("Days to Maintenance", f"{days_to_maintenance}")

        # Maintenance recommendation
        if failure_prob > 0.7:
            st.error("⚠️ Immediate maintenance required!")
        elif failure_prob > 0.4:
            st.warning("⚠️ Schedule maintenance soon")
        else:
            st.success("✅ Equipment operating normally")

        # Equipment maintenance schedule
        if len(components['equipment']) > 0:
            st.subheader("Maintenance Schedule")
            for name, component in components['equipment'].items():
                schedule = component.schedule_maintenance()
                if schedule:
                    st.write(f"{name} Maintenance:")
                    for task in schedule:
                        st.write(f"- Month {task['month']}: {task['type'].title()} "
                                 f"(${task['cost']:,.2f})")


if __name__ == "__main__":
    main()v