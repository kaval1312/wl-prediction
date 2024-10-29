# src/utils/plotting.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any


def create_production_plot(
        df: pd.DataFrame,
        show_water_cut: bool = True,
        show_gas: bool = False,
        show_pressure: bool = False
) -> go.Figure:
    """Create production profile visualization."""
    n_subplots = 1 + sum([show_water_cut, show_gas, show_pressure])
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Oil Production"] +
                       (["Water Cut"] if show_water_cut else []) +
                       (["Gas Production"] if show_gas else []) +
                       (["Reservoir Pressure"] if show_pressure else [])
    )

    # Oil production
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df['Production'],
            name='Oil Production',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )

    current_row = 1

    if show_water_cut and 'Water_Cut' in df.columns:
        current_row += 1
        fig.add_trace(
            go.Scatter(
                x=df['Month'],
                y=df['Water_Cut'] * 100,
                name='Water Cut %',
                line=dict(color='blue', width=2)
            ),
            row=current_row, col=1
        )

    fig.update_layout(
        height=250 * n_subplots,
        showlegend=True,
        title_text="Production Profile",
        template="plotly_white"
    )

    return fig


def create_costs_plot(
        df: pd.DataFrame,
        cost_categories: List[str],
        cumulative: bool = False
) -> go.Figure:
    """Create cost analysis visualization."""
    fig = go.Figure()

    if cumulative:
        for category in cost_categories:
            fig.add_trace(
                go.Scatter(
                    x=df['Month'],
                    y=df[category].cumsum(),
                    name=category.replace('_', ' ').title(),
                    stackgroup='one',
                    line=dict(width=0)
                )
            )
    else:
        for category in cost_categories:
            fig.add_trace(
                go.Bar(
                    x=df['Month'],
                    y=df[category],
                    name=category.replace('_', ' ').title()
                )
            )
        fig.update_layout(barmode='stack')

    fig.update_layout(
        title_text="Cost Analysis",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        showlegend=True,
        template="plotly_white",
        height=400
    )

    return fig


def create_cash_flow_plot(
        df: pd.DataFrame,
        include_npv: bool = True
) -> go.Figure:
    """Create cash flow visualization."""
    fig = go.Figure()

    # Monthly cash flow bars
    fig.add_trace(
        go.Bar(
            x=df['Month'],
            y=df['Net_Revenue'],
            name='Monthly Cash Flow',
            marker_color=np.where(df['Net_Revenue'] >= 0, 'green', 'red')
        )
    )

    # Cumulative cash flow line
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df['Net_Revenue'].cumsum(),
            name='Cumulative Cash Flow',
            line=dict(color='blue', width=2)
        )
    )

    fig.update_layout(
        title_text="Cash Flow Analysis",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        showlegend=True,
        template="plotly_white",
        height=400
    )

    return fig


def create_monte_carlo_plots(results: Dict) -> List[go.Figure]:
    """Create Monte Carlo simulation visualizations."""
    figures = []

    # Production uncertainty cone
    fig_prod = go.Figure()

    if 'production_profiles' in results:
        p10_line = np.percentile(results['production_profiles'], 10, axis=0)
        p90_line = np.percentile(results['production_profiles'], 90, axis=0)
        mean_line = np.mean(results['production_profiles'], axis=0)

        x_range = np.arange(len(p10_line))

        fig_prod.add_trace(go.Scatter(
            x=x_range,
            y=p90_line,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,255,0.2)',
            name='P90'
        ))

        fig_prod.add_trace(go.Scatter(
            x=x_range,
            y=p10_line,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,255,0.2)',
            name='P10'
        ))

        fig_prod.add_trace(go.Scatter(
            x=x_range,
            y=mean_line,
            line=dict(color='blue', width=2),
            name='Mean'
        ))

        fig_prod.update_layout(
            title="Production Uncertainty",
            xaxis_title="Month",
            yaxis_title="Rate (bbl/d)",
            template="plotly_white"
        )
        figures.append(fig_prod)

    # NPV distribution
    if 'npv_distribution' in results:
        fig_npv = go.Figure()
        fig_npv.add_trace(go.Histogram(
            x=results['npv_distribution'],
            nbinsx=50,
            name='NPV Distribution',
            marker_color='blue',
            opacity=0.7
        ))

        fig_npv.update_layout(
            title="NPV Distribution",
            xaxis_title="NPV ($)",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        figures.append(fig_npv)

    return figures


def create_equipment_health_plot(
        df: pd.DataFrame,
        show_maintenance: bool = True
) -> go.Figure:
    """Create equipment health visualization."""
    fig = make_subplots(
        rows=2 if show_maintenance else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15
    )

    # Health score line
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df['Reliability'] * 100,
            name='Health Score',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Add reference lines
    fig.add_hline(y=90, line_dash="dash", line_color="green",
                  annotation_text="Excellent", row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="yellow",
                  annotation_text="Warning", row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="Critical", row=1, col=1)

    if show_maintenance and 'Maintenance_Event' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[df['Maintenance_Event'] == 1]['Month'],
                y=[100] * len(df[df['Maintenance_Event'] == 1]),
                mode='markers',
                name='Maintenance',
                marker=dict(symbol='triangle-up', size=12, color='red')
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=600 if show_maintenance else 400,
        title_text="Equipment Health Monitoring",
        showlegend=True,
        template="plotly_white"
    )

    return fig

def create_abandonment_plots(abandonment_data: Dict[str, Any]) -> List[go.Figure]:
    # Extract relevant data from the abandonment_data dictionary
    years = abandonment_data.get('years', [])
    costs = abandonment_data.get('costs', [])
    cumulative_costs = abandonment_data.get('cumulative_costs', [])
    
    # Create the cost breakdown plot
    cost_breakdown_fig = go.Figure()
    cost_breakdown_fig.add_trace(go.Bar(x=years, y=costs, name='Annual Cost'))
    cost_breakdown_fig.add_trace(go.Scatter(x=years, y=cumulative_costs, name='Cumulative Cost', yaxis='y2'))
    
    cost_breakdown_fig.update_layout(
        title='Abandonment Cost Breakdown',
        xaxis_title='Year',
        yaxis_title='Annual Cost ($)',
        yaxis2=dict(title='Cumulative Cost ($)', overlaying='y', side='right'),
        legend_title='Cost Type'
    )
    
    # Create the timeline plot
    timeline_fig = go.Figure()
    timeline_fig.add_trace(go.Scatter(x=years, y=costs, mode='lines+markers', name='Abandonment Cost'))
    
    timeline_fig.update_layout(
        title='Abandonment Cost Timeline',
        xaxis_title='Year',
        yaxis_title='Cost ($)',
        showlegend=False
    )
    
    return [cost_breakdown_fig, timeline_fig]


def create_tornado_plot(
        sensitivity_results: Dict[str, List[tuple]],
        base_case: float
) -> go.Figure:
    """Create tornado diagram for sensitivity analysis."""
    fig = go.Figure()

    # Calculate percentage changes
    changes = []
    for var_name, results in sensitivity_results.items():
        min_val = min(r[1] for r in results)
        max_val = max(r[1] for r in results)
        changes.append({
            'variable': var_name,
            'low_change': (min_val - base_case) / base_case * 100,
            'high_change': (max_val - base_case) / base_case * 100
        })

    # Sort by total impact
    changes.sort(key=lambda x: abs(x['high_change'] - x['low_change']), reverse=True)

    for change in changes:
        fig.add_trace(go.Bar(
            y=[change['variable']],
            x=[change['high_change'] - change['low_change']],
            base=change['low_change'],
            orientation='h',
            name=change['variable']
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title_text="Sensitivity Analysis",
        xaxis_title="Percentage Change from Base Case",
        showlegend=False,
        template="plotly_white",
        height=400
    )

    return fig


def create_technical_plots(technical_data: Dict[str, Any]) -> List[go.Figure]:
    figures = []

    # Create a figure for each technical parameter
    for param, values in technical_data.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode='lines', name=param))
        fig.update_layout(title=f'{param} Over Time', xaxis_title='Time', yaxis_title=param)
        figures.append(fig)

    return figures