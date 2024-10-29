import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

def create_production_plot(
    df: pd.DataFrame,
    show_water_cut: bool = True,
    show_gas: bool = False,
    show_pressure: bool = False
) -> go.Figure:
    """
    Create production profile visualization.
    
    Args:
        df: DataFrame with production data
        show_water_cut: Whether to show water cut
        show_gas: Whether to show gas production
        show_pressure: Whether to show reservoir pressure
    
    Returns:
        Plotly figure object
    """
    n_subplots = 1 + sum([show_water_cut, show_gas, show_pressure])
    fig = make_subplots(rows=n_subplots, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.08)
    
    # Oil production
    fig.add_trace(
        go.Scatter(x=df['Month'],
                  y=df['Production'] * (1 - df['Water_Cut']),
                  name='Oil Production',
                  line=dict(color='green')),
        row=1, col=1
    )
    
    current_row = 1
    
    if show_water_cut:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df['Month'],
                      y=df['Water_Cut'] * 100,
                      name='Water Cut %',
                      line=dict(color='blue')),
            row=current_row, col=1
        )
    
    if show_gas:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df['Month'],
                      y=df['Gas_Production'],
                      name='Gas Production',
                      line=dict(color='red')),
            row=current_row, col=1
        )
    
    if show_pressure:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=df['Month'],
                      y=df['Reservoir_Pressure'],
                      name='Pressure (psi)',
                      line=dict(color='purple')),
            row=current_row, col=1
        )
    
    fig.update_layout(height=200*n_subplots,
                     showlegend=True,
                     title_text="Production Profile")
    
    return fig

def create_costs_plot(
    df: pd.DataFrame,
    cost_categories: List[str],
    cumulative: bool = False
) -> go.Figure:
    """
    Create cost analysis visualization.
    
    Args:
        df: DataFrame with cost data
        cost_categories: List of cost column names
        cumulative: Whether to show cumulative costs
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if cumulative:
        for category in cost_categories:
            fig.add_trace(
                go.Scatter(x=df['Month'],
                          y=df[category].cumsum(),
                          name=category.replace('_', ' ').title(),
                          stackgroup='one')
            )
    else:
        for category in cost_categories:
            fig.add_trace(
                go.Bar(x=df['Month'],
                      y=df[category],
                      name=category.replace('_', ' ').title())
            )
        fig.update_layout(barmode='stack')
    
    fig.update_layout(
        title_text="Cost Analysis",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        showlegend=True
    )
    
    return fig

def create_cash_flow_plot(
    df: pd.DataFrame,
    include_npv: bool = True
) -> go.Figure:
    """
    Create cash flow visualization.
    
    Args:
        df: DataFrame with financial data
        include_npv: Whether to show NPV line
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Monthly cash flow bars
    fig.add_trace(
        go.Bar(x=df['Month'],
               y=df['Net_Revenue'],
               name='Monthly Cash Flow',
               marker_color=np.where(df['Net_Revenue'] >= 0, 'green', 'red'))
    )
    
    # Cumulative cash flow line
    fig.add_trace(
        go.Scatter(x=df['Month'],
                  y=df['Net_Revenue'].cumsum(),
                  name='Cumulative Cash Flow',
                  line=dict(color='blue'))
    )
    
    if include_npv and 'NPV' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Month'],
                      y=df['NPV'],
                      name='NPV',
                      line=dict(color='purple', dash='dash'))
        )
    
    fig.update_layout(
        title_text="Cash Flow Analysis",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        showlegend=True
    )
    
    return fig

def create_sensitivity_plot(
    sensitivity_results: Dict[str, List[Tuple[float, float]]],
    base_case: float,
    tornado: bool = True
) -> go.Figure:
    """
    Create sensitivity analysis visualization.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        base_case: Base case value
        tornado: Whether to create tornado diagram
    
    Returns:
        Plotly figure object
    """
    if tornado:
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
        changes.sort(key=lambda x: abs(x['high_change'] - x['low_change']),
                    reverse=True)
        
        fig = go.Figure()
        
        for change in changes:
            fig.add_trace(
                go.Bar(
                    y=[change['variable']],
                    x=[change['high_change'] - change['low_change']],
                    base=change['low_change'],
                    orientation='h',
                    name=change['variable']
                )
            )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title_text="Sensitivity Analysis (Tornado Diagram)",
            xaxis_title="Percentage Change from Base Case",
            showlegend=False
        )
    
    else:
        fig = go.Figure()
        
        for var_name, results in sensitivity_results.items():
            x_vals, y_vals = zip(*results)
            fig.add_trace(
                go.Scatter(x=x_vals,
                          y=y_vals,
                          name=var_name)
            )
        
        fig.add_hline(y=base_case, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title_text="Sensitivity Analysis",
            xaxis_title="Variable Value",
            yaxis_title="Output Metric",
            showlegend=True
        )
    
    return fig

def create_equipment_health_plot(
    df: pd.DataFrame,
    equipment_components: Dict,
    show_maintenance: bool = True
) -> go.Figure:
    """
    Create equipment health visualization.
    
    Args:
        df: DataFrame with equipment data
        equipment_components: Dictionary of equipment components
        show_maintenance: Whether to show maintenance events
    
    Returns:
        Plotly figure object
    """
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.1,
                       subplot_titles=("Equipment Health", "Maintenance Events"))
    
    # Equipment health lines
    for component in equipment_components.values():
        failure_prob = component.calculate_failure_probability(len(df))
        health = (1 - failure_prob) * 100
        
        fig.add_trace(
            go.Scatter(x=df['Month'],
                      y=health,
                      name=f"{component.name} Health",
                      line=dict(dash='solid')),
            row=1, col=1
        )
    
    if show_maintenance:
        # Maintenance events
        maintenance_events = df['Maintenance_Events'] > 0
        fig.add_trace(
            go.Scatter(x=df.loc[maintenance_events, 'Month'],
                      y=[100] * sum(maintenance_events),
                      mode='markers',
                      name='Maintenance',
                      marker=dict(symbol='triangle-up',
                                size=10,
                                color='red')),
            row=2, col=1
        )
    
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_layout(height=600, showlegend=True,
                     title_text="Equipment Health Analysis")
    
    return fig