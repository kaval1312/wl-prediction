# src/utils/plotting.py

from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from datetime import datetime
from .monte_carlo import MonteCarloResults


def create_production_profile(df: pd.DataFrame, show_water_cut: bool = True) -> go.Figure:
    """Create production profile visualization."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df['Production'],
            name='Oil Production',
            line=dict(color='green', width=2)
        ),
        secondary_y=False
    )

    if show_water_cut and 'Water_Cut' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Month'],
                y=df['Water_Cut'] * 100,
                name='Water Cut',
                line=dict(color='blue', width=2)
            ),
            secondary_y=True
        )

    fig.update_layout(
        title_text="Production Profile",
        template="plotly_white",
        hovermode='x unified',
        showlegend=True
    )

    fig.update_yaxes(title_text="Production (bbl/d)", secondary_y=False)
    fig.update_yaxes(title_text="Water Cut (%)", secondary_y=True)

    return fig


def create_cost_analysis(df: pd.DataFrame, categories: List[str]) -> go.Figure:
    """Create cost analysis visualization."""
    fig = go.Figure()

    for category in categories:
        fig.add_trace(
            go.Bar(
                x=df['Month'],
                y=df[category],
                name=category.replace('_', ' ').title()
            )
        )

    fig.update_layout(
        title="Cost Analysis",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        barmode='stack',
        template="plotly_white",
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def create_cashflow_analysis(df: pd.DataFrame) -> go.Figure:
    """Create cash flow visualization."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df['Month'],
            y=df['Net_Revenue'],
            name='Monthly Cash Flow',
            marker_color=np.where(df['Net_Revenue'] >= 0, 'green', 'red')
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df['Cumulative_Cash_Flow'],
            name='Cumulative Cash Flow',
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )

    fig.update_layout(
        title_text="Cash Flow Analysis",
        template="plotly_white",
        hovermode='x unified',
        showlegend=True
    )

    fig.update_yaxes(title_text="Monthly Cash Flow ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Cash Flow ($)", secondary_y=True)

    return fig


def create_monte_carlo_analysis(results: MonteCarloResults) -> List[go.Figure]:
    """Create Monte Carlo simulation visualizations."""
    figures = []

    # Production uncertainty cone
    fig_prod = go.Figure()

    production_profiles = results.production_profiles
    p10_line = np.percentile(production_profiles, 10, axis=0)
    p90_line = np.percentile(production_profiles, 90, axis=0)
    p50_line = np.percentile(production_profiles, 50, axis=0)
    mean_line = np.mean(production_profiles, axis=0)

    x_range = np.arange(len(p10_line))

    # P90-P10 range
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
        y=p50_line,
        line=dict(color='blue', width=2, dash='dash'),
        name='P50'
    ))

    fig_prod.add_trace(go.Scatter(
        x=x_range,
        y=mean_line,
        line=dict(color='red', width=2),
        name='Mean'
    ))

    fig_prod.update_layout(
        title="Production Uncertainty",
        xaxis_title="Month",
        yaxis_title="Rate (bbl/d)",
        template="plotly_white",
        hovermode='x unified'
    )
    figures.append(fig_prod)

    # NPV distribution
    fig_npv = go.Figure()

    kde = gaussian_kde(results.npv_distribution)
    x_range = np.linspace(min(results.npv_distribution),
                          max(results.npv_distribution), 100)

    fig_npv.add_trace(go.Histogram(
        x=results.npv_distribution,
        nbinsx=50,
        name='NPV Distribution',
        marker_color='rgba(0,100,255,0.5)',
        showlegend=False
    ))

    fig_npv.add_trace(go.Scatter(
        x=x_range,
        y=kde(x_range) * (max(results.npv_distribution) -
                          min(results.npv_distribution)) / 50,
        mode='lines',
        line=dict(color='red', width=2),
        name='Distribution Curve'
    ))

    fig_npv.update_layout(
        title="NPV Distribution",
        xaxis_title="NPV ($)",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    figures.append(fig_npv)

    return figures


def create_risk_analysis(results: MonteCarloResults) -> go.Figure:
    """Create visualization of risk metrics."""
    fig = go.Figure()

    metrics = [
        ('Probability of Loss', results.risk_metrics['probability_of_loss'] * 100),
        ('Probability of Target ROI', results.risk_metrics['probability_of_target_roi'] * 100),
        ('Value at Risk ($M)', results.risk_metrics['value_at_risk'] / 1e6),
        ('Expected Shortfall ($M)', results.risk_metrics['expected_shortfall'] / 1e6)
    ]

    fig.add_trace(go.Bar(
        x=[m[0] for m in metrics],
        y=[m[1] for m in metrics],
        marker_color=['red' if 'Loss' in m[0] else 'blue' for m in metrics],
        text=[f"{m[1]:.1f}{'%' if 'Probability' in m[0] else 'M'}" for m in metrics],
        textposition='auto',
    ))

    fig.update_layout(
        title="Risk Metrics Summary",
        xaxis_title="Metric",
        yaxis_title="Value",
        template="plotly_white",
        showlegend=False
    )

    return fig


def create_sensitivity_analysis(
        results: MonteCarloResults,
        base_npv: float,
        parameter_ranges: Dict[str, Tuple[float, float]]
) -> go.Figure:
    """Create tornado diagram for sensitivity analysis."""
    fig = go.Figure()

    impacts = []
    for param, (low, high) in parameter_ranges.items():
        low_impact = (low - base_npv) / base_npv * 100
        high_impact = (high - base_npv) / base_npv * 100
        impacts.append({
            'parameter': param,
            'low': low_impact,
            'high': high_impact,
            'range': abs(high_impact - low_impact)
        })

    impacts.sort(key=lambda x: x['range'], reverse=True)

    for impact in impacts:
        fig.add_trace(go.Bar(
            name=impact['parameter'],
            y=[impact['parameter']],
            x=[impact['high'] - impact['low']],
            base=impact['low'],
            orientation='h',
            marker_color='lightblue'
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Sensitivity Analysis",
        xaxis_title="Change in NPV (%)",
        yaxis_title="Parameter",
        template="plotly_white",
        showlegend=False
    )

    return fig


def create_correlation_matrix(results: MonteCarloResults) -> go.Figure:
    """Create correlation matrix of key metrics."""
    data = pd.DataFrame({
        'Production': results.production_profiles[:, -1],
        'NPV': results.npv_distribution,
        'ROI': results.roi_distribution,
        'Payback': results.payback_distribution
    })

    fig = go.Figure(data=go.Splom(
        dimensions=[
            dict(label='Production', values=data['Production']),
            dict(label='NPV', values=data['NPV']),
            dict(label='ROI', values=data['ROI']),
            dict(label='Payback', values=data['Payback'])
        ],
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(
            size=4,
            color='blue',
            opacity=0.5,
            showscale=False
        )
    ))

    fig.update_layout(
        title="Metric Correlations",
        template="plotly_white",
        height=800,
        width=800,
        showlegend=False,
        dragmode='select'
    )

    return fig


def create_probability_analysis(results: MonteCarloResults) -> go.Figure:
    """Create cumulative probability plot for NPV and ROI distributions."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("NPV Cumulative Probability",
                                        "ROI Cumulative Probability"))

    sorted_npv = np.sort(results.npv_distribution)
    sorted_roi = np.sort(results.roi_distribution)
    probs = np.linspace(0, 100, len(sorted_npv))

    fig.add_trace(
        go.Scatter(
            x=sorted_npv,
            y=probs,
            mode='lines',
            name='NPV',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sorted_roi,
            y=probs,
            mode='lines',
            name='ROI',
            line=dict(color='green')
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Cumulative Probability Distributions",
        template="plotly_white",
        height=500,
        showlegend=False
    )

    fig.update_xaxes(title_text="NPV ($)", row=1, col=1)
    fig.update_xaxes(title_text="ROI (%)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability (%)", row=1, col=2)

    return fig


def create_abandonment_analysis(
        costs: Dict[str, float],
        timeline: pd.DataFrame
) -> List[go.Figure]:
    """Create abandonment analysis visualizations."""
    figures = []

    fig_costs = go.Figure(data=[go.Pie(
        labels=list(costs.keys()),
        values=list(costs.values()),
        hole=.3
    )])

    fig_costs.update_layout(
        title="Abandonment Cost Breakdown",
        template="plotly_white"
    )
    figures.append(fig_costs)

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=timeline['Date'],
        y=timeline['Cumulative_Cost'],
        mode='lines+markers',
        name='Cumulative Cost'
    ))

    fig_timeline.update_layout(
        title="Abandonment Cost Timeline",
        xaxis_title="Date",
        yaxis_title="Cumulative Cost ($)",
        template="plotly_white"
    )
    figures.append(fig_timeline)

    return figures


def create_technical_analysis(params: Dict[str, Any]) -> List[go.Figure]:
    """Create technical analysis visualizations."""
    figures = []

    # Pressure-Volume-Temperature plot
    fig_pvt = go.Figure()

    pressures = np.linspace(0, params['reservoir_pressure'], 50)
    temperatures = np.linspace(0, params['temperature'], 50)

    X, Y = np.meshgrid(pressures, temperatures)
    Z = X * Y / (params['reservoir_pressure'] * params['temperature'])

    fig_pvt.add_trace(go.Surface(x=X, y=Y, z=Z))

    fig_pvt.update_layout(
        title="PVT Analysis",
        scene=dict(
            xaxis_title="Pressure (psi)",
            yaxis_title="Temperature (°F)",
            zaxis_title="Volume Factor"
        ),
        template="plotly_white"
    )
    figures.append(fig_pvt)

    # Pressure vs. Time plot
    fig_pressure = go.Figure()

    time = np.linspace(0, 12, 50)  # 12 months
    pressure_decline = params['reservoir_pressure'] * np.exp(-0.1 * time)

    fig_pressure.add_trace(go.Scatter(
        x=time,
        y=pressure_decline,
        mode='lines',
        name='Reservoir Pressure'
    ))

    fig_pressure.update_layout(
        title="Reservoir Pressure Decline",
        xaxis_title="Time (months)",
        yaxis_title="Pressure (psi)",
        template="plotly_white"
    )
    figures.append(fig_pressure)

    return figures


def create_equipment_analysis(df: pd.DataFrame) -> go.Figure:
    """Create equipment health and performance visualization."""
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Equipment Health", "Maintenance Events"))

    # Health score line
    fig.add_trace(
        go.Scatter(
            x=df['Month'],
            y=df.get('Health_Score', np.ones(len(df))) * 100,
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

    # Maintenance events
    if 'Maintenance_Events' in df.columns:
        maintenance_months = df[df['Maintenance_Events'] == 1]['Month']
        fig.add_trace(
            go.Scatter(
                x=maintenance_months,
                y=[1] * len(maintenance_months),
                mode='markers',
                name='Maintenance',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='red'
                )
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        showlegend=True,
        title_text="Equipment Performance Analysis"
    )

    # Update axes labels
    fig.update_yaxes(title_text="Health Score (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Events", range=[0, 2], row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)

    return fig


def create_monthly_analysis(results: MonteCarloResults) -> go.Figure:
    """Create monthly metrics analysis visualization."""
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Production", "NPV", "ROI"))

    months = np.arange(results.production_profiles.shape[1])

    # Production plot
    p10_prod = np.percentile(results.production_profiles, 10, axis=0)
    p50_prod = np.percentile(results.production_profiles, 50, axis=0)
    p90_prod = np.percentile(results.production_profiles, 90, axis=0)

    fig.add_trace(
        go.Scatter(
            x=months,
            y=p50_prod,
            name='P50 Production',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=p10_prod,
            name='P10 Production',
            line=dict(color='lightblue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=p90_prod,
            name='P90 Production',
            line=dict(color='lightblue', dash='dash')
        ),
        row=1, col=1
    )

    # NPV plot
    npv_cumulative = np.cumsum(np.sort(results.npv_distribution))
    fig.add_trace(
        go.Scatter(
            x=months[:len(npv_cumulative)],
            y=npv_cumulative,
            name='Cumulative NPV',
            line=dict(color='green')
        ),
        row=2, col=1
    )

    # ROI plot
    roi_monthly = results.roi_distribution / len(months)
    fig.add_trace(
        go.Scatter(
            x=months,
            y=roi_monthly,
            name='Monthly ROI',
            line=dict(color='red')
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        template="plotly_white",
        showlegend=True,
        title_text="Monthly Performance Metrics"
    )

    # Update axes labels
    fig.update_yaxes(title_text="Production (bbl/d)", row=1, col=1)
    fig.update_yaxes(title_text="NPV ($)", row=2, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=3, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=1)

    return fig