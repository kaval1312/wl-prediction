import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

# Import from project structure
from src.calculations.production import calculate_decline_curve as calc_decline
from src.calculations.costs import calculate_operating_costs, calculate_maintenance_costs
from src.calculations.npv import calculate_npv
from src.calculations.taxes import calculate_tax_obligations

from src.models.equipment import EquipmentComponent
from src.models.financial import FinancialMetrics

from src.utils.plotting import create_production_plot
from src.utils.monte_carlo import MonteCarloSimulator

# Configure page
st.set_page_config(
    page_title="Oil Well Prediction",
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
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_decline_curve(initial_rate, decline_rate, months, b_factor=0):
    """Calculate production decline curve using project module"""
    return calc_decline(
        initial_rate=initial_rate,
        decline_rate=decline_rate,
        months=months,
        b_factor=b_factor
    )

def predict_equipment_failure(vibration, temperature, health):
    """Predict equipment failure using project models"""
    equipment = EquipmentComponent(
        name="ESP",
        expected_life=60,
        replacement_cost=50000,
        failure_impact=0.8,
        maintenance_schedule=3,
        operating_conditions=health/100
    )
    
    parameters = {
        'vibration': vibration,
        'temperature': temperature
    }
    
    reliability = equipment.calculate_reliability(
        age=12,  # months
        parameters=parameters,
        maintenance_history=[]
    )
    
    return 1 - reliability

def calculate_economics(production, oil_price, opex, project_data):
    """Calculate economic metrics using project modules"""
    # Calculate operating costs
    costs, cost_components = calculate_operating_costs(
        production=production,
        water_cut=np.zeros_like(production),
        base_cost=opex
    )
    
    # Calculate revenue and cash flow
    revenue = production * oil_price
    cash_flow = revenue - costs
    
    # Calculate NPV
    npv = calculate_npv(
        cash_flows=cash_flow,
        discount_rate=0.1
    )
    
    # Calculate taxes if tax data is available
    if 'tax_data' in project_data:
        tax_rates = {'federal': 0.21, 'state': 0.05, 'severance': 0.045}
        taxes, _ = calculate_tax_obligations(revenue, costs, tax_rates, {})
    else:
        taxes = np.zeros_like(revenue)
    
    return {
        'revenue': revenue,
        'costs': costs,
        'cash_flow': cash_flow,
        'npv': npv,
        'total_revenue': revenue.sum(),
        'total_costs': costs.sum(),
        'taxes': taxes,
        'cost_components': cost_components
    }

def main():
    st.title("🛢️ Oil Well Production Analysis")
    
    # Load project data
    project_data = load_project_data()
    if project_data is None:
        st.error("Failed to load project data. Using default values.")
        project_data = {}
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Production parameters
    st.sidebar.subheader("Production")
    initial_rate = st.sidebar.number_input("Initial Rate (bbl/day)", value=1000.0, min_value=0.0)
    decline_rate = st.sidebar.number_input("Annual Decline Rate (%)", value=15.0, min_value=0.0, max_value=100.0) / 100 / 12
    b_factor = st.sidebar.slider("Hyperbolic Decline Factor", 0.0, 1.0, 0.0)
    forecast_months = st.sidebar.slider("Forecast Months", 12, 120, 60)
    
    # Economic parameters
    st.sidebar.subheader("Economics")
    oil_price = st.sidebar.number_input("Oil Price ($/bbl)", value=70.0, min_value=0.0)
    opex = st.sidebar.number_input("Operating Cost ($/bbl)", value=20.0, min_value=0.0)
    
    # Equipment parameters
    st.sidebar.subheader("Equipment")
    current_health = st.sidebar.slider("Equipment Health (%)", 0, 100, 85)
    pump_vibration = st.sidebar.slider("Pump Vibration", 0.0, 1.0, 0.2)
    motor_temperature = st.sidebar.slider("Motor Temperature (°F)", 150, 250, 180)
    
    # Calculate production forecast
    production = calculate_decline_curve(initial_rate, decline_rate, forecast_months, b_factor)
    time = np.arange(forecast_months)
    
    # Calculate economics
    economics = calculate_economics(production, oil_price, opex, project_data)
    
    # Calculate equipment reliability
    failure_prob = predict_equipment_failure(pump_vibration, motor_temperature, current_health)
    
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
            f"${economics['cash_flow'][0]:,.0f}/d",
            f"${economics['cash_flow'][1] - economics['cash_flow'][0]:,.0f}"
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
            f"{failure_prob*100:.1f}%",
            "Critical" if failure_prob > 0.7 else "Normal"
        )
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["Production", "Economics", "Equipment"])
    
    with tab1:
        st.subheader("Production Forecast")
        fig = create_production_plot(time, production)
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
    
    with tab2:
        st.subheader("Economic Analysis")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time,
            y=economics['revenue'],
            name='Revenue',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=time,
            y=economics['costs'],
            name='Costs',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=time,
            y=economics['cash_flow'],
            name='Cash Flow',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            xaxis_title="Months",
            yaxis_title="Amount ($/d)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Economic metrics
        st.subheader("Economic Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NPV", f"${economics['npv']:,.0f}")
        with col2:
            st.metric("Total Revenue", f"${economics['total_revenue']:,.0f}")
        with col3:
            st.metric("Total Costs", f"${economics['total_costs']:,.0f}")
    
    with tab3:
        st.subheader("Equipment Analysis")
        
        # Equipment health gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_health,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Equipment Health"},
            gauge = {
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
            st.metric("Failure Probability", f"{failure_prob*100:.1f}%")
            days_to_maintenance = max(0, int((0.7 - failure_prob) / 0.01))
            st.metric("Days to Maintenance", f"{days_to_maintenance}")
        
        # Maintenance recommendation
        if failure_prob > 0.7:
            st.error("⚠️ Immediate maintenance required!")
        elif failure_prob > 0.4:
            st.warning("⚠️ Schedule maintenance soon")
        else:
            st.success("✅ Equipment operating normally")

if __name__ == "__main__":
    main()