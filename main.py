
import streamlit as st
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from models.equipment import EquipmentComponent
from models.environmental import EnvironmentalRegulation
from models.financial import TaxCalculator

from calculations.production import (
    calculate_decline_curve,
    calculate_water_cut_increase
)
from calculations.costs import (
    calculate_maintenance_costs,
    calculate_personnel_costs,
    calculate_insurance_costs
)
from calculations.taxes import calculate_tax_obligations
from calculations.npv import calculate_npv

from utils.plotting import create_analysis_plots
from utils.helpers import load_config, validate_inputs

class OilWellCalculator:
    """Main application class for the Oil Well Abandonment Calculator."""
    
    def __init__(self):
        """Initialize the calculator with configuration settings."""
        self.config_path = Path(__file__).parent / 'config'
        self.load_configurations()
        self.initialize_components()

    def load_configurations(self):
        """Load configuration files for equipment, regulations, and tax rates."""
        self.equipment_config = load_config(self.config_path / 'equipment.yaml')
        self.regulations_config = load_config(self.config_path / 'regulations.yaml')
        self.tax_config = load_config(self.config_path / 'tax_rates.yaml')

    def initialize_components(self):
        """Initialize equipment and regulatory components."""
        self.equipment_components = {
            name: EquipmentComponent(**specs)
            for name, specs in self.equipment_config.items()
        }
        
        self.environmental_regulations = {
            name: EnvironmentalRegulation(**specs)
            for name, specs in self.regulations_config.items()
        }
        
        self.tax_calculator = TaxCalculator(self.tax_config)

    def run_streamlit_app(self):
        """Run the main Streamlit application."""
        st.title("Oil Well Abandonment Calculator")
        self.setup_sidebar()
        self.run_calculations()
        self.display_results()

    def setup_sidebar(self):
        """Setup the Streamlit sidebar with input parameters."""
        st.sidebar.header("Production Parameters")
        self.inputs = {
            'initial_production': st.sidebar.number_input(
                "Initial Production (barrels/month)",
                value=1000.0, min_value=0.0
            ),
            'decline_rate': st.sidebar.number_input(
                "Annual Decline Rate (%)",
                value=15.0, min_value=0.0, max_value=100.0
            ) / 100 / 12,
            'b_factor': st.sidebar.slider(
                "Decline Curve b-factor",
                0.0, 1.0, 0.0
            )
        }
        
        # Add equipment configuration inputs
        st.sidebar.header("Equipment Configuration")
        for name, component in self.equipment_components.items():
            component.operating_conditions = st.sidebar.slider(
                f"{name.title()} Operating Conditions",
                0.0, 1.0, 0.8
            )

    def run_calculations(self):
        """Execute all calculations for the analysis."""
        try:
            validate_inputs(self.inputs)
            self.calculate_production()
            self.calculate_costs()
            self.calculate_revenue()
            self.calculate_metrics()
        except ValueError as e:
            st.error(f"Calculation Error: {str(e)}")
            return False
        return True

    def calculate_production(self):
        """Calculate production profiles and decline curves."""
        months = 120  # 10-year analysis
        self.production = calculate_decline_curve(
            self.inputs['initial_production'],
            self.inputs['decline_rate'],
            months,
            self.inputs['b_factor']
        )
        self.water_cut, self.pressure = calculate_water_cut_increase(
            months,
            initial_water_cut=0.1,
            max_water_cut=0.95,
            reservoir_pressure=3000,
            pressure_decline_rate=0.1
        )

    def calculate_costs(self):
        """Calculate all associated costs."""
        months = len(self.production)
        
        # Equipment costs
        self.equipment_costs = np.zeros(months)
        self.equipment_failures = np.zeros(months)
        for component in self.equipment_components.values():
            failure_prob = component.calculate_failure_probability(months)
            failures = np.random.binomial(1, failure_prob)
            self.equipment_failures += failures
            self.equipment_costs += failures * component.replacement_cost
        
        # Environmental costs
        self.environmental_costs = np.zeros(months)
        for regulation in self.environmental_regulations.values():
            self.environmental_costs += regulation.calculate_monthly_cost(months)
        
        # Operational costs
        self.maintenance_costs, self.maintenance_events = calculate_maintenance_costs(
            months=months,
            base_cost=5000,
            inflation_rate=0.03,
            equipment_age=0,
            well_depth=5000
        )

    def calculate_revenue(self):
        """Calculate revenue and tax obligations."""
        oil_price = 70  # $/barrel
        self.gross_revenue = self.production * (1 - self.water_cut) * oil_price
        self.total_costs = (
            self.maintenance_costs +
            self.equipment_costs +
            self.environmental_costs
        )
        
        self.tax_obligations = np.array([
            self.tax_calculator.calculate_obligations(
                revenue=self.gross_revenue[i],
                costs=self.total_costs[i]
            )
            for i in range(len(self.production))
        ])
        
        self.net_revenue = self.gross_revenue - self.total_costs - self.tax_obligations

    def calculate_metrics(self):
        """Calculate key performance metrics."""
        self.npv = calculate_npv(self.net_revenue)
        self.create_dataframe()
        self.calculate_risk_metrics()

    def create_dataframe(self):
        """Create a DataFrame with all calculated values."""
        self.df = pd.DataFrame({
            'Month': range(1, len(self.production) + 1),
            'Production': self.production,
            'Water_Cut': self.water_cut,
            'Gross_Revenue': self.gross_revenue,
            'Equipment_Costs': self.equipment_costs,
            'Environmental_Costs': self.environmental_costs,
            'Maintenance_Costs': self.maintenance_costs,
            'Tax_Obligations': self.tax_obligations,
            'Net_Revenue': self.net_revenue
        })

    def display_results(self):
        """Display analysis results in Streamlit."""
        if not hasattr(self, 'df'):
            return
            
        create_analysis_plots(
            df=self.df,
            equipment_components=self.equipment_components,
            environmental_regulations=self.environmental_regulations
        )
        
        self.display_metrics()
        self.display_recommendations()

    def display_metrics(self):
        """Display key metrics and analysis results."""
        st.header("Economic Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Net Present Value", f"${self.npv:,.2f}")
        with col2:
            st.metric("Total Oil Production", 
                     f"{(self.production * (1 - self.water_cut)).sum():,.0f} bbls")
        with col3:
            st.metric("Average Monthly Net Revenue",
                     f"${self.net_revenue.mean():,.2f}")

    def display_recommendations(self):
        """Display abandonment recommendations."""
        st.header("Abandonment Recommendations")
        
        # Calculate abandonment score
        economic_score = max(0, min(100, (self.net_revenue.mean() / 10000) * 100))
        equipment_score = max(0, min(100, (1 - np.mean(self.equipment_failures)) * 100))
        
        overall_score = (economic_score + equipment_score) / 2
        st.progress(overall_score/100)
        st.write(f"Overall Well Health Score: {overall_score:.1f}/100")
        
        if overall_score < 30:
            st.error("Recommendation: Consider immediate abandonment")
        elif overall_score < 60:
            st.warning("Recommendation: Monitor closely and prepare abandonment plan")
        else:
            st.success("Recommendation: Continue operations with regular monitoring")

def main():
    """Main entry point for the Streamlit application."""
    calculator = OilWellCalculator()
    calculator.run_streamlit_app()

if __name__ == "__main__":
    main()

