```python
# src/visualization/visualizer.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from ..core.data_structures import WellData
from datetime import datetime

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self):
        """Initialize visualizer with default settings"""
        self.theme = {
            'background': '#ffffff',
            'text': '#000000',
            'grid': '#e5e5e5',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }
        self.default_height = 600
        self.default_width = 1000

    def plot_production_profile(self, well_data: WellData) -> go.Figure:
        """Create production profile visualization"""
        try:
            prod_data = well_data.production_data

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Oil & Gas Production', 'Water Cut'),
                vertical_spacing=0.15
            )

            # Oil and Gas Production
            fig.add_trace(
                go.Scatter(
                    x=prod_data['Date'],
                    y=prod_data['Oil_Production_BBL'],
                    name='Oil Production (BBL)',
                    line=dict(color=self.theme['primary'])
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=prod_data['Date'],
                    y=prod_data['Gas_Production_MCF'],
                    name='Gas Production (MCF)',
                    line=dict(color=self.theme['secondary']),
                    yaxis='y2'
                ),
                row=1, col=1
            )

            # Water Cut
            fig.add_trace(
                go.Scatter(
                    x=prod_data['Date'],
                    y=prod_data['Water_Cut_Percentage'],
                    name='Water Cut (%)',
                    line=dict(color='red')
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                height=self.default_height,
                width=self.default_width,
                title_text=f"Production Profile - {well_data.well_name}",
                showlegend=True,
                hovermode='x unified'
            )

            # Update y-axes
            fig.update_yaxes(title_text="Oil Production (BBL)", row=1, col=1)
            fig.update_yaxes(title_text="Gas Production (MCF)",
                             overlaying='y', side='right', row=1, col=1)
            fig.update_yaxes(title_text="Water Cut (%)", row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"Error creating production profile: {str(e)}")
            return go.Figure()

    def plot_decline_curves(self, well_data: WellData,
                            decline_results: Dict) -> go.Figure:
        """Create decline curve visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Oil Decline', 'Gas Decline'),
                vertical_spacing=0.15
            )

            # Actual production data
            prod_data = well_data.production_data

            # Oil decline
            fig.add_trace(
                go.Scatter(
                    x=prod_data['Date'],
                    y=prod_data['Oil_Production_BBL'],
                    name='Actual Oil Production',
                    mode='markers',
                    marker=dict(color=self.theme['primary'])
                ),
                row=1, col=1
            )

            if 'oil_decline_curve' in decline_results:
                fig.add_trace(
                    go.Scatter(
                        x=decline_results['forecast_dates'],
                        y=decline_results['oil_decline_curve'],
                        name='Oil Decline Forecast',
                        line=dict(color=self.theme['primary'], dash='dash')
                    ),
                    row=1, col=1
                )

            # Gas decline
            fig.add_trace(
                go.Scatter(
                    x=prod_data['Date'],
                    y=prod_data['Gas_Production_MCF'],
                    name='Actual Gas Production',
                    mode='markers',
                    marker=dict(color=self.theme['secondary'])
                ),
                row=2, col=1
            )

            if 'gas_decline_curve' in decline_results:
                fig.add_trace(
                    go.Scatter(
                        x=decline_results['forecast_dates'],
                        y=decline_results['gas_decline_curve'],
                        name='Gas Decline Forecast',
                        line=dict(color=self.theme['secondary'], dash='dash')
                    ),
                    row=2, col=1
                )

            # Update layout
            fig.update_layout(
                height=self.default_height,
                width=self.default_width,
                title_text=f"Decline Curve Analysis - {well_data.well_name}",
                showlegend=True,
                hovermode='x unified'
            )

            # Update axes
            fig.update_yaxes(title_text="Oil Production (BBL)", row=1, col=1)
            fig.update_yaxes(title_text="Gas Production (MCF)", row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"Error creating decline curves: {str(e)}")
            return go.Figure()

    def plot_monte_carlo_results(self, mc_results: Dict) -> Dict[str, go.Figure]:
        """Create Monte Carlo simulation visualizations"""
        try:
            figures = {}

            # Production Monte Carlo
            if 'production' in mc_results:
                prod_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(
                        'Oil Production Distribution',
                        'Gas Production Distribution',
                        'Water Cut Distribution'
                    ),
                    vertical_spacing=0.15
                )

                for i, (key, data) in enumerate(mc_results['production'].items()):
                    prod_fig.add_trace(
                        go.Histogram(
                            x=data,
                            name=key,
                            nbinsx=50,
                            histnorm='probability density'
                        ),
                        row=i + 1, col=1
                    )

                prod_fig.update_layout(
                    height=900,
                    width=self.default_width,
                    title_text="Production Monte Carlo Results",
                    showlegend=True
                )

                figures['production'] = prod_fig

            # Economic Monte Carlo
            if 'economics' in mc_results:
                econ_fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'NPV Distribution',
                        'Revenue Distribution',
                        'OPEX Distribution',
                        'Cash Flow Distribution'
                    )
                )

                metrics = ['npv', 'revenue', 'opex', 'cash_flow']
                positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

                for metric, pos in zip(metrics, positions):
                    if metric in mc_results['economics']:
                        econ_fig.add_trace(
                            go.Histogram(
                                x=mc_results['economics'][metric],
                                name=metric.upper(),
                                nbinsx=50,
                                histnorm='probability density'
                            ),
                            row=pos[0], col=pos[1]
                        )

                econ_fig.update_layout(
                    height=800,
                    width=self.default_width,
                    title_text="Economic Monte Carlo Results",
                    showlegend=True
                )

                figures['economics'] = econ_fig

            return figures

        except Exception as e:
            logger.error(f"Error creating Monte Carlo visualizations: {str(e)}")
            return {}

    def plot_economic_analysis(self, economic_results: Dict) -> go.Figure:
        """Create economic analysis visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Cumulative Cash Flow',
                    'Monthly Revenue vs Costs',
                    'Cost Breakdown',
                    'Sensitivity Analysis'
                )
            )

            # Cumulative Cash Flow
            fig.add_trace(
                go.Scatter(
                    x=range(len(economic_results['cumulative_cash_flows'])),
                    y=economic_results['cumulative_cash_flows'],
                    name='Cumulative Cash Flow'
                ),
                row=1, col=1
            )

            # Monthly Revenue vs Costs
            fig.add_trace(
                go.Bar(
                    x=['Revenue', 'OPEX', 'Maintenance'],
                    y=[
                        economic_results['cumulative_revenue'],
                        economic_results['cumulative_opex'],
                        economic_results['cumulative_maintenance']
                    ],
                    name='Monthly Metrics'
                ),
                row=1, col=2
            )

            # Cost Breakdown
            if 'cost_categories' in economic_results:
                fig.add_trace(
                    go.Pie(
                        labels=list(economic_results['cost_categories'].keys()),
                        values=list(economic_results['cost_categories'].values()),
                        name='Cost Breakdown'
                    ),
                    row=2, col=1
                )

            # Sensitivity Analysis
            if 'sensitivity' in economic_results:
                sensitivities = economic_results['sensitivity']
                fig.add_trace(
                    go.Waterfall(
                        name='Sensitivity',
                        orientation='h',
                        measure=['relative'] * len(sensitivities),
                        x=list(sensitivities.values()),
                        y=list(sensitivities.keys())
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                height=800,
                width=self.default_width,
                title_text="Economic Analysis Results",
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating economic analysis visualization: {str(e)}")
            return go.Figure()

    def plot_environmental_metrics(self, env_results: Dict) -> go.Figure:
        """Create environmental metrics visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Emissions Trends',
                    'Water Management',
                    'Environmental Costs',
                    'Compliance Metrics'
                )
            )

            # Emissions Trends
            if 'emissions' in env_results:
                for emission_type, values in env_results['emissions'].items():
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(values))),
                            y=values,
                            name=emission_type
                        ),
                        row=1, col=1
                    )

            # Water Management
            if 'water' in env_results:
                fig.add_trace(
                    go.Bar(
                        x=['Produced', 'Treated', 'Recycled'],
                        y=[
                            env_results['water'].get('total_produced_water', 0),
                            env_results['water'].get('total_treated_water', 0),
                            env_results['water'].get('total_recycled_water', 0)
                        ],
                        name='Water Management'
                    ),
                    row=1, col=2
                )

            # Environmental Costs
            if 'costs' in env_results:
                fig.add_trace(
                    go.Pie(
                        labels=list(env_results['costs']['category_costs'].keys()),
                        values=list(env_results['costs']['category_costs'].values()),
                        name='Environmental Costs'
                    ),
                    row=2, col=1
                )

            # Compliance Metrics
            if 'compliance' in env_results:
                fig.add_trace(
                    go.Indicator(
                        mode='gauge+number',
                        value=env_results['compliance'].get('overall_compliance', 0) * 100,
                        title={'text': 'Overall Compliance %'},
                        gauge={'axis': {'range': [0, 100]}}
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                height=800,
                width=self.default_width,
                title_text="Environmental Metrics",
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating environmental visualization: {str(e)}")
            return go.Figure()

    def plot_maintenance_analysis(self, maint_results: Dict) -> go.Figure:
        """Create maintenance analysis visualization"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Maintenance Events Timeline',
                    'Cost Distribution by Category',
                    'Equipment Reliability (MTBF)',
                    'Priority Distribution',
                    'Cost Trends',
                    'Predicted vs Actual'
                )
            )

            # Maintenance Events Timeline
            if 'events' in maint_results:
                events = pd.DataFrame(maint_results['events'])
                for category in events['Maintenance_Category'].unique():
                    cat_events = events[events['Maintenance_Category'] == category]
                    fig.add_trace(
                        go.Scatter(
                            x=cat_events['Date'],
                            y=cat_events['Cost'],
                            mode='markers',
                            name=category,
                            marker=dict(size=10),
                            hovertemplate=(
                                    "<b>Date:</b> %{x}<br>" +
                                    "<b>Cost:</b> $%{y:,.2f}<br>" +
                                    "<b>Category:</b> " + category +
                                    "<extra></extra>"
                            )
                        ),
                        row=1, col=1
                    )

            # Cost Distribution by Category
            if 'cost_breakdown' in maint_results:
                costs = maint_results['cost_breakdown']
                fig.add_trace(
                    go.Pie(
                        labels=list(costs.keys()),
                        values=list(costs.values()),
                        name='Cost Distribution',
                        hovertemplate=(
                                "<b>Category:</b> %{label}<br>" +
                                "<b>Cost:</b> $%{value:,.2f}<br>" +
                                "<b>Percentage:</b> %{percent}<extra></extra>"
                        )
                    ),
                    row=1, col=2
                )

            # Equipment Reliability (MTBF)
            if 'reliability' in maint_results:
                reliability = maint_results['reliability']
                fig.add_trace(
                    go.Bar(
                        x=list(reliability.keys()),
                        y=[r.get('mtbf_days', 0) for r in reliability.values()],
                        name='MTBF (days)',
                        hovertemplate=(
                                "<b>Equipment:</b> %{x}<br>" +
                                "<b>MTBF:</b> %{y:.1f} days<extra></extra>"
                        )
                    ),
                    row=2, col=1
                )

            # Priority Distribution
            if 'priority_distribution' in maint_results:
                priorities = maint_results['priority_distribution']
                fig.add_trace(
                    go.Bar(
                        x=list(priorities.keys()),
                        y=list(priorities.values()),
                        name='Priority Distribution',
                        marker_color=['red', 'yellow', 'green'],
                        hovertemplate=(
                                "<b>Priority:</b> %{x}<br>" +
                                "<b>Count:</b> %{y}<extra></extra>"
                        )
                    ),
                    row=2, col=2
                )

            # Cost Trends
            if 'cost_trends' in maint_results:
                trends = pd.DataFrame(maint_results['cost_trends'])
                fig.add_trace(
                    go.Scatter(
                        x=trends.index,
                        y=trends['rolling_mean'],
                        name='30-Day Moving Average',
                        line=dict(color='blue'),
                        hovertemplate=(
                                "<b>Date:</b> %{x}<br>" +
                                "<b>Average Cost:</b> $%{y:,.2f}<extra></extra>"
                        )
                    ),
                    row=3, col=1
                )

                # Add confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=trends.index,
                        y=trends['rolling_mean'] + trends['rolling_std'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,255,0.2)',
                        showlegend=False
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=trends.index,
                        y=trends['rolling_mean'] - trends['rolling_std'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,255,0.2)',
                        name='±1 Std Dev'
                    ),
                    row=3, col=1
                )

            # Predicted vs Actual Maintenance
            if 'predictions' in maint_results and 'actual' in maint_results:
                pred_vs_actual = pd.DataFrame({
                    'Predicted': maint_results['predictions'].get('predicted_costs', []),
                    'Actual': maint_results['actual'].get('actual_costs', [])
                })
                fig.add_trace(
                    go.Scatter(
                        x=pred_vs_actual.index,
                        y=pred_vs_actual['Predicted'],
                        name='Predicted Costs',
                        line=dict(dash='dash'),
                        hovertemplate=(
                                "<b>Date:</b> %{x}<br>" +
                                "<b>Predicted Cost:</b> $%{y:,.2f}<extra></extra>"
                        )
                    ),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=pred_vs_actual.index,
                        y=pred_vs_actual['Actual'],
                        name='Actual Costs',
                        hovertemplate=(
                                "<b>Date:</b> %{x}<br>" +
                                "<b>Actual Cost:</b> $%{y:,.2f}<extra></extra>"
                        )
                    ),
                    row=3, col=2
                )

            # Update layout
            fig.update_layout(
                height=1200,
                width=self.default_width,
                title_text="Maintenance Analysis Dashboard",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Update axes labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Cost ($)", row=1, col=1)

            fig.update_xaxes(title_text="Equipment", row=2, col=1)
            fig.update_yaxes(title_text="MTBF (days)", row=2, col=1)

            fig.update_xaxes(title_text="Priority Level", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)

            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Cost ($)", row=3, col=1)

            fig.update_xaxes(title_text="Date", row=3, col=2)
            fig.update_yaxes(title_text="Cost ($)", row=3, col=2)

            return fig

        except Exception as e:
            logger.error(f"Error creating maintenance visualization: {str(e)}")
            return go.Figure()
