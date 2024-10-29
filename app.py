# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Oil Well Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        .stProgress > div > div > div > div {
            background-color: #00ff00;
        }
    </style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load or generate sample well data"""
    data = {
        'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'oil_production': [1000, 950, 902, 857, 814, 773, 734, 697, 662, 629, 598, 568],
        'water_cut': [0.1, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45],
        'pressure': [2000, 1950, 1900, 1850, 1800, 1750, 1700, 1650, 1600, 1550, 1500, 1450],
        'equipment_health': [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78],
        'pump_vibration': [0.1, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45],
        'motor_temperature': [150, 152, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178],
        'maintenance_cost': [5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200]
    }
    return pd.DataFrame(data)

def predict_equipment_failure(vibration, temperature, health):
    """Predict equipment failure probability using multiple indicators"""
    # Normalize inputs
    vib_norm = (vibration - 0.1) / 0.35  # Normalize to 0-1 range
    temp_norm = (temperature - 150) / 50  # Normalize to 0-1 range
    health_norm = health / 100
    
    # Calculate failure probability using weighted average
    weights = [0.4, 0.3, 0.3]  # Weights for vibration, temperature, and health
    failure_prob = (
        vib_norm * weights[0] + 
        temp_norm * weights[1] + 
        (1 - health_norm) * weights[2]
    )
    
    return min(max(failure_prob, 0), 1)  # Ensure probability is between 0 and 1

def calculate_economic_metrics(production, oil_price, opex, capex, tax_rate=0.2):
    """Calculate comprehensive economic metrics"""
    revenue = production * oil_price
    operating_costs = production * opex
    taxable_income = revenue - operating_costs
    taxes = taxable_income * tax_rate
    net_cash_flow = revenue - operating_costs - taxes
    
    # Calculate NPV
    discount_rate = 0.1
    periods = np.arange(len(production))
    npv = sum(net_cash_flow / (1 + discount_rate) ** periods)
    
    # Calculate ROI
    total_investment = capex + operating_costs.sum()
    roi = (revenue.sum() - total_investment) / total_investment * 100
    
    # Calculate Payback Period
    cumulative_cash_flow = np.cumsum(net_cash_flow)
    payback_period = np.where(cumulative_cash_flow >= capex)[0][0] if any(cumulative_cash_flow >= capex) else len(production)
    
    return {
        'npv': npv,
        'roi': roi,
        'payback_period': payback_period,
        'net_cash_flow': net_cash_flow,
        'taxes': taxes,
        'cumulative_cash_flow': cumulative_cash_flow
    }

def create_visualizations(historical_data, forecast_data, economic_metrics):
    """Create comprehensive visualizations"""
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Production Profile',
            'Economic Analysis',
            'Equipment Health Indicators',
            'Cash Flow Analysis',
            'Failure Probability',
            'Sensitivity Analysis'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Production Profile
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['oil_production'],
            name='Historical Production',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['production'],
            name='Forecast Production',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # Economic Analysis
    fig.add_trace(
        go.Bar(
            x=['NPV', 'ROI', 'Payback'],
            y=[
                economic_metrics['npv'],
                economic_metrics['roi'],
                economic_metrics['payback_period']
            ],
            name='Economic Metrics'
        ),
        row=1, col=2
    )
    
    # Equipment Health
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['equipment_health'],
            name='Equipment Health',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['pump_vibration'],
            name='Pump Vibration',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    # Cash Flow
    fig.add_trace(
        go.Scatter(
            x=forecast_data['date'],
            y=economic_metrics['cumulative_cash_flow'],
            name='Cumulative Cash Flow',
            fill='tozeroy'
        ),
        row=2, col=2
    )
    
    # Failure Probability
    failure_probs = [
        predict_equipment_failure(v, t, h) 
        for v, t, h in zip(
            historical_data['pump_vibration'],
            historical_data['motor_temperature'],
            historical_data['equipment_health']
        )
    ]
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=failure_probs,
            name='Failure Probability',
            line=dict(color='red')
        ),
        row=3, col=1
    )
    
    # Sensitivity Analysis
    oil_prices = np.linspace(30, 100, 10)
    npvs = []
    for price in oil_prices:
        metrics = calculate_economic_metrics(
            forecast_data['production'],
            price,
            opex=20,
            capex=1000000
        )
        npvs.append(metrics['npv'])
    
    fig.add_trace(
        go.Scatter(
            x=oil_prices,
            y=npvs,
            name='NPV Sensitivity'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Comprehensive Well Analysis"
    )
    
    return fig

def main():
    st.title("Advanced Oil Well Analysis")
    
    # Sidebar inputs
    st.sidebar.header("Production Parameters")
    initial_rate = st.sidebar.number_input("Initial Rate (bbl/day)", value=1000.0)
    decline_rate = st.sidebar.number_input("Decline Rate (%/year)", value=15.0) / 100 / 12
    forecast_months = st.sidebar.slider("Forecast Months", 12, 120, 60)
    
    st.sidebar.header("Economic Parameters")
    oil_price = st.sidebar.number_input("Oil Price ($/bbl)", value=70.0)
    opex = st.sidebar.number_input("Operating Cost ($/bbl)", value=20.0)
    capex = st.sidebar.number_input("Capital Cost ($)", value=1000000.0)
    tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 50, 20) / 100
    
    # Load data
    historical_data = load_sample_data()
    
    # Calculate forecast
    forecast_dates = pd.date_range(
        start=historical_data['date'].iloc[-1] + pd.DateOffset(months=1),
        periods=forecast_months,
        freq='M'
    )
    forecast_production = np.array([
        initial_rate * (1 - decline_rate) ** i 
        for i in range(forecast_months)
    ])
    
    forecast_data = pd.DataFrame({
        'date': forecast_dates,
        'production': forecast_production
    })
    
    # Calculate economic metrics
    economic_metrics = calculate_economic_metrics(
        forecast_production,
        oil_price,
        opex,
        capex,
        tax_rate
    )
    
    # Create and display visualizations
    fig = create_visualizations(historical_data, forecast_data, economic_metrics)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed metrics
    st.header("Detailed Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Economic Metrics")
        st.metric("NPV", f"${economic_metrics['npv']:,.0f}")
        st.metric("ROI", f"{economic_metrics['roi']:.1f}%")
        st.metric("Payback Period", f"{economic_metrics['payback_period']:.1f} months")
    
    with col2:
        st.subheader("Production Metrics")
        st.metric("Current Production", f"{historical_data['oil_production'].iloc[-1]:.0f} bbl/day")
        st.metric("Decline Rate", f"{decline_rate*12*100:.1f}%/year")
        st.metric("Cumulative Production", f"{forecast_production.sum():,.0f} bbl")
    
    with col3:
        st.subheader("Equipment Health")
        failure_prob = predict_equipment_failure(
            historical_data['pump_vibration'].iloc[-1],
            historical_data['motor_temperature'].iloc[-1],
            historical_data['equipment_health'].iloc[-1]
        )
        st.metric("Equipment Health", f"{historical_data['equipment_health'].iloc[-1]:.0f}%")
        st.metric("Failure Probability", f"{failure_prob*100:.1f}%")
        st.metric("Days to Maintenance", 
                 f"{max(0, int((0.7 - failure_prob) / 0.01))}")
    
    # Maintenance recommendations
    st.header("Maintenance Recommendations")
    if failure_prob > 0.7:
        st.error("⚠️ Immediate maintenance required!")
    elif failure_prob > 0.4:
        st.warning("⚠️ Schedule maintenance soon")
    else:
        st.success("✅ Equipment operating normally")
    
    # Download data
    st.header("Download Analysis")
    combined_data = pd.concat([
        historical_data,
        forecast_data.set_index('date')
    ]).reset_index()
    
    st.download_button(
        label="Download Complete Analysis",
        data=combined_data.to_csv(index=False),
        file_name="well_analysis.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()