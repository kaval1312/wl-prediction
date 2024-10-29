import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import MonteCarloSimulator, EconomicAnalyzer, EquipmentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Oil Well Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'monte_carlo' not in st.session_state:
    st.session_state.monte_carlo = MonteCarloSimulator()
if 'economic' not in st.session_state:
    st.session_state.economic = EconomicAnalyzer()
if 'equipment' not in st.session_state:
    st.session_state.equipment = EquipmentAnalyzer()

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
</style>
""", unsafe_allow_html=True)

def create_mock_data():
    """Create mock data for demonstration"""
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
        }
    }

def main():
    # Load mock data
    data = create_mock_data()
    
    # Dashboard Header
    st.title("🛢️ Oil Well Analysis Dashboard")
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current Production",
            f"{data['production']['oil_rate'].iloc[-1]:.0f} bbl/d",
            f"{data['production']['oil_rate'].diff().iloc[-1]:.1f} bbl/d"
        )
    with col2:
        st.metric(
            "Water Cut",
            f"{data['production']['water_cut'].iloc[-1]*100:.1f}%",
            f"{data['production']['water_cut'].diff().iloc[-1]*100:.1f}%"
        )
    with col3:
        st.metric(
            "Gas Rate",
            f"{data['production']['gas_rate'].iloc[-1]:.0f} mcf/d",
            f"{data['production']['gas_rate'].diff().iloc[-1]:.1f} mcf/d"
        )
    with col4:
        st.metric(
            "Pressure",
            f"{data['production']['pressure'].iloc[-1]:.0f} psi",
            f"{data['production']['pressure'].diff().iloc[-1]:.1f} psi"
        )
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["Production", "Economics", "Equipment"])
    
    with tab1:
        st.subheader("Production Overview")
        
        # Production plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['production']['date'],
            y=data['production']['oil_rate'],
            name='Oil Rate',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=data['production']['date'],
            y=data['production']['water_cut'] * data['production']['oil_rate'],
            name='Water Rate',
            line=dict(color='blue')
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis_title="Rate (bbl/d)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Economic Analysis")
        
        # Economic inputs
        col1, col2 = st.columns(2)
        with col1:
            oil_price = st.number_input("Oil Price ($/bbl)", value=70.0)
            opex = st.number_input("Operating Cost ($/bbl)", value=20.0)
        with col2:
            initial_investment = st.number_input("Initial Investment ($)", value=1000000.0)
        
        # Calculate economics
        production = data['production']['oil_rate'].values
        scenarios = st.session_state.economic.calculate_scenarios(
            production, oil_price, opex, initial_investment
        )
        
        # Display economic metrics
        for scenario, metrics in scenarios.items():
            st.write(f"**{scenario.title()} Scenario**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NPV", f"${metrics['npv']:,.0f}")
            with col2:
                st.metric("IRR", f"{metrics['irr']*100:.1f}%" if metrics['irr'] else "N/A")
            with col3:
                st.metric("Net Cash Flow", f"${metrics['net_cash_flow']:,.0f}")
    
    with tab3:
        st.subheader("Equipment Analysis")
        
        # Equipment parameters
        equipment_data = {
            'pump': {
                'vibration': data['equipment']['pump_vibration'],
                'temperature': data['equipment']['motor_temperature']
            }
        }
        
        # Calculate reliability
        reliability, violations = st.session_state.equipment.calculate_reliability(
            'pump',
            equipment_data['pump'],
            age=180,  # 6 months
            maintenance_history=[data['equipment']['last_maintenance']]
        )
        
        # Predict failure
        needs_maintenance, days_until_critical = st.session_state.equipment.predict_failure(
            reliability,
            equipment_data['pump']
        )
        
        # Display equipment metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Equipment Reliability", f"{reliability*100:.1f}%")
            st.metric("Days Until Maintenance", days_until_critical)
        with col2:
            st.metric("Vibration Level", f"{data['equipment']['pump_vibration']:.2f}")
            st.metric("Motor Temperature", f"{data['equipment']['motor_temperature']}°F")
        
        if violations:
            st.warning("Parameter Violations:")
            for violation in violations:
                st.write(f"- {violation}")
        
        if needs_maintenance:
            st.error("⚠️ Maintenance Required!")
            st.write(f"Schedule maintenance within {days_until_critical} days")
    
    # Recent Alerts
    st.header("Recent Alerts")
    alerts = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='H'),
        'type': ['Warning', 'Info', 'Error', 'Info', 'Warning'],
        'message': [
            'High pump vibration detected',
            'Daily production target achieved',
            'Equipment maintenance overdue',
            'Water cut within normal range',
            'Pressure trending lower'
        ]
    })
    
    for _, alert in alerts.iterrows():
        if alert['type'] == 'Error':
            st.error(f"{alert['timestamp']}: {alert['message']}")
        elif alert['type'] == 'Warning':
            st.warning(f"{alert['timestamp']}: {alert['message']}")
        else:
            st.info(f"{alert['timestamp']}: {alert['message']}")

if __name__ == "__main__":
    main()
