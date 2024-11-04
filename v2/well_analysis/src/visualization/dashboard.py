# src/visualization/dashboard.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import logging
from datetime import datetime
from ..core.data_structures import WellData
from ..core.data_loader import DataLoader
from ..compute.local_compute import LocalCompute
from ..compute.hpc_manager import HPCManager
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class Dashboard:
    def __init__(self):
        """Initialize dashboard"""
        self.visualizer = Visualizer()
        self.local_compute = LocalCompute()
        self.hpc_manager = HPCManager()

    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Well Analysis Dashboard",
            page_icon="🛢️",
            layout="wide"
        )

        self._render_sidebar()
        self._render_main_content()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("Well Analysis Dashboard")

        # Data Loading Section
        st.sidebar.header("Data Loading")
        data_dir = st.sidebar.text_input(
            "Data Directory",
            value=str(Path.cwd() / "data")
        )

        if st.sidebar.button("Load Data"):
            self.data_loader = DataLoader(data_dir)
            self.data_loader.load_all_data()
            st.session_state.wells_data = self.data_loader.wells_data
            st.success("Data loaded successfully!")

        # Well Selection
        if 'wells_data' in st.session_state:
            selected_well = st.sidebar.selectbox(
                "Select Well",
                options=list(st.session_state.wells_data.keys())
            )
            st.session_state.selected_well = selected_well

        # Analysis Controls
        st.sidebar.header("Analysis Controls")
        compute_option = st.sidebar.radio(
            "Compute Option",
            options=["Local", "HPC"]
        )

        if compute_option == "HPC":
            st.sidebar.text_input("HPC Host")
            st.sidebar.text_input("Username")
            st.sidebar.text_input("SSH Key Path")

        # Analysis Parameters
        st.sidebar.subheader("Analysis Parameters")
        n_simulations = st.sidebar.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000
        )

        forecast_days = st.sidebar.slider(
            "Forecast Days",
            min_value=30,
            max_value=3650,
            value=365
        )

        if st.sidebar.button("Run Analysis"):
            self._run_analysis(
                compute_option,
                n_simulations,
                forecast_days
            )

    def _run_analysis(self, compute_option: str,
                      n_simulations: int,
                      forecast_days: int):
        """Run selected analyses"""
        try:
            well_data = st.session_state.wells_data[st.session_state.selected_well]

            if compute_option == "Local":
                results = self.local_compute.run_integrated_analysis(
                    well_data,
                    {
                        'n_simulations': n_simulations,
                        'forecast_days': forecast_days
                    }
                )
            else:
                results = self.hpc_manager.run_analysis(
                    'integrated_analysis',
                    {
                        'well_data': well_data,
                        'n_simulations': n_simulations,
                        'forecast_days': forecast_days
                    }
                )

            st.session_state.analysis_results = results
            st.success("Analysis completed successfully!")

        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")

    def _render_main_content(self):
        """Render main dashboard content"""
        st.title("Well Analysis Dashboard")

        if 'selected_well' not in st.session_state:
            st.info("Please load data and select a well from the sidebar.")
            return

        # Create tabs for different views
        tabs = st.tabs([
            "Production",
            "Economics",
            "Environmental",
            "Maintenance",
            "Monte Carlo",
            "Summary"
        ])

        well_data = st.session_state.wells_data[st.session_state.selected_well]

        # Production Tab
        with tabs[0]:
            st.header("Production Analysis")
            fig = self.visualizer.plot_production_profile(well_data)
            st.plotly_chart(fig, use_container_width=True)

            if 'analysis_results' in st.session_state and 'decline' in st.session_state.analysis_results:
                fig = self.visualizer.plot_decline_curves(
                    well_data,
                    st.session_state.analysis_results['decline']
                )
                st.plotly_chart(fig, use_container_width=True)

        # Economics Tab
        with tabs[1]:
            st.header("Economic Analysis")
            if 'analysis_results' in st.session_state and 'economic' in st.session_state.analysis_results:
                fig = self.visualizer.plot_economic_analysis(
                    st.session_state.analysis_results['economic']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run analysis to view economic results")

        # Environmental Tab
        with tabs[2]:
            st.header("Environmental Analysis")
            if 'analysis_results' in st.session_state and 'environmental' in st.session_state.analysis_results:
                fig = self.visualizer.plot_environmental_metrics(
                    st.session_state.analysis_results['environmental']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run analysis to view environmental results")

        # Maintenance Tab
        with tabs[3]:
            st.header("Maintenance Analysis")
            if 'analysis_results' in st.session_state and 'maintenance' in st.session_state.analysis_results:
                fig = self.visualizer.plot_maintenance_analysis(
                    st.session_state.analysis_results['maintenance']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run analysis to view maintenance results")

        # Monte Carlo Tab
        with tabs[4]:
            st.header("Monte Carlo Analysis")
            if 'analysis_results' in st.session_state and 'monte_carlo' in st.session_state.analysis_results:
                figs = self.visualizer.plot_monte_carlo_results(
                    st.session_state.analysis_results['monte_carlo']
                )
                for name, fig in figs.items():
                    st.subheader(name.capitalize())
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run analysis to view Monte Carlo results")

        # Summary Tab
        with tabs[5]:
            st.header("Summary Dashboard")
            if 'analysis_results' in st.session_state:
                fig = self.visualizer.create_summary_dashboard(
                    well_data,
                    st.session_state.analysis_results
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run analysis to view summary dashboard")


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()