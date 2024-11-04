import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Union, Tuple
import scipy.stats as stats
import networkx as nx
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class WellConfig:
    basin: str
    type: str
    depth: float
    crew_size: int
    contractor: str
    lease_terms: str
    initial_cost: float
    environmental_sensitivity: str
    water_source: str
    proximity_to_protected_areas: str
    soil_type: str
    emission_control_system: str
    spill_prevention_system: str
    groundwater_monitoring: str


class WellAnalysisCore:
    def __init__(self):
        self.monte_carlo_params = {
            'Production': {
                'Oil_Rate': {'min': 0.7, 'most_likely': 1.0, 'max': 1.3},
                'Gas_Rate': {'min': 0.6, 'most_likely': 1.0, 'max': 1.4},
                'Water_Cut': {'min': 0.8, 'most_likely': 1.0, 'max': 1.2}
            },
            'Costs': {
                'Operating_Cost': {'min': 0.8, 'most_likely': 1.0, 'max': 1.3},
                'Maintenance_Cost': {'min': 0.7, 'most_likely': 1.0, 'max': 1.5}
            },
            'Prices': {
                'Oil_Price': {'min': 40, 'most_likely': 70, 'max': 100},
                'Gas_Price': {'min': 2, 'most_likely': 3.5, 'max': 5}
            },
            'Environmental': {
                'Emission_Rates': {'min': 0.8, 'most_likely': 1.0, 'max': 1.2},
                'Water_Treatment_Cost': {'min': 0.9, 'most_likely': 1.0, 'max': 1.4}
            }
        }

        self.maintenance_items = {
            'Replacement_Parts': {
                'ESP_System': {
                    'Pump': 15000, 'Motor': 12000, 'Cable': 5000,
                    'Sensor_Package': 3000, 'Controller': 4000, 'VFD': 8000
                },
                'Wellhead_Equipment': {
                    'Christmas_Tree': 20000, 'Valves': 5000,
                    'Pressure_Gauges': 1000, 'Safety_Devices': 3000,
                    'Flanges': 2000, 'Spools': 1500
                },
                'Downhole_Equipment': {
                    'Tubing': 10000, 'Packers': 8000, 'Safety_Valves': 6000,
                    'Gas_Lift_Valves': 4000, 'Production_Liner': 15000,
                    'Sand_Screens': 7000
                },
                'Surface_Equipment': {
                    'Separators': 25000, 'Heater_Treaters': 20000,
                    'Storage_Tanks': 30000, 'Compressors': 15000,
                    'Pumping_Units': 18000
                }
            },
            'Services': {
                'Regular_Maintenance': {
                    'Well_Logging': 5000, 'Pressure_Testing': 3000,
                    'Flow_Testing': 2500, 'Corrosion_Monitoring': 2000,
                    'Scale_Treatment': 1500
                },
                'Cleaning': {
                    'Tank_Cleaning': 4000, 'Pipeline_Pigging': 3500,
                    'Well_Bore_Cleaning': 5000, 'Surface_Equipment_Cleaning': 2000
                },
                'Inspection': {
                    'NDT_Inspection': 3000, 'Corrosion_Inspection': 2500,
                    'Safety_System_Check': 2000, 'Environmental_Compliance_Check': 2500
                }
            },
            'Repairs': {
                'Emergency_Repairs': {
                    'Leak_Repair': 10000, 'Equipment_Failure': 15000,
                    'Power_System': 8000, 'Control_System': 5000
                },
                'Scheduled_Repairs': {
                    'Worn_Parts': 5000, 'Calibration': 2000,
                    'Preventive_Replacement': 4000
                }
            }
        }

        self.environmental_categories = {
            'Emissions_Monitoring': {
                'Methane_Emissions': {
                    'Continuous_Monitoring': 1000, 'Leak_Detection': 800,
                    'Repair_Program': 1200, 'Reporting': 500
                },
                'CO2_Emissions': {
                    'Monitoring': 900, 'Reporting': 400, 'Offset_Programs': 2000
                },
                'VOC_Emissions': {
                    'Monitoring': 700, 'Control_Systems': 1500, 'Reporting': 300
                }
            },
            'Water_Management': {
                'Produced_Water': {
                    'Treatment': 2000, 'Disposal': 1500,
                    'Recycling': 2500, 'Quality_Testing': 800
                },
                'Groundwater': {
                    'Monitoring_Wells': 1200, 'Sample_Analysis': 600,
                    'Report_Generation': 400, 'Remediation': 5000
                },
                'Surface_Water': {
                    'Quality_Monitoring': 700, 'Runoff_Control': 900,
                    'Spill_Prevention': 1000
                }
            },
            'Soil_Management': {
                'Contamination_Monitoring': {
                    'Sampling': 500, 'Analysis': 800, 'Reporting': 300
                },
                'Remediation': {
                    'Soil_Treatment': 3000, 'Disposal': 2000,
                    'Site_Restoration': 5000
                }
            },
            'Waste_Management': {
                'Drilling_Waste': {
                    'Treatment': 1500, 'Disposal': 2000, 'Recycling': 1000
                },
                'Chemical_Waste': {
                    'Storage': 800, 'Treatment': 1200, 'Disposal': 1500
                },
                'General_Waste': {
                    'Collection': 300, 'Sorting': 200, 'Disposal': 500
                }
            },
            'Wildlife_Protection': {
                'Habitat_Preservation': {
                    'Monitoring': 1000, 'Protection_Measures': 1500,
                    'Restoration': 3000
                },
                'Species_Monitoring': {
                    'Surveys': 800, 'Reporting': 400, 'Mitigation': 2000
                }
            }
        }

        self.financial_categories = {
            'Labor': {
                'Operations': {
                    'Operators': 5000, 'Technicians': 4500, 'Engineers': 7000,
                    'Supervisors': 8000, 'Support_Staff': 3500
                },
                'Maintenance': {
                    'Maintenance_Crew': 4000, 'Specialists': 6000,
                    'Contractors': 5500
                },
                'Management': {
                    'Site_Manager': 10000, 'HSE_Manager': 8000,
                    'Technical_Manager': 9000
                }
            },
            'Insurance': {
                'Operational': {
                    'Equipment_Insurance': 2000, 'Business_Interruption': 3000,
                    'General_Liability': 2500
                },
                'Environmental': {
                    'Pollution_Liability': 4000, 'Environmental_Damage': 3500,
                    'Remediation_Cost': 3000
                },
                'Personnel': {
                    'Workers_Comp': 1500, 'Health_Insurance': 2000,
                    'Life_Insurance': 1000
                }
            },
            'Lease': {
                'Land': {
                    'Surface_Rights': 10000, 'Mineral_Rights': 15000,
                    'Access_Rights': 5000
                },
                'Equipment': {
                    'Heavy_Machinery': 8000, 'Vehicles': 3000,
                    'Temporary_Equipment': 2000
                },
                'Facilities': {
                    'Office_Space': 2000, 'Storage': 1500,
                    'Worker_Facilities': 1000
                }
            },
            'Regulatory_Compliance': {
                'Permits': {
                    'Drilling_Permits': 5000, 'Environmental_Permits': 4000,
                    'Operating_Permits': 3000
                },
                'Reporting': {
                    'Environmental_Reports': 2000, 'Production_Reports': 1500,
                    'Safety_Reports': 1000
                },
                'Audits': {
                    'Environmental_Audits': 3000, 'Safety_Audits': 2500,
                    'Financial_Audits': 4000
                }
            }
        }

    def run_monte_carlo(self, params: Dict, n_simulations: int = 1000) -> Dict[str, np.ndarray]:
        results = {}
        for category, items in params.items():
            category_results = {}
            for item, ranges in items.items():
                if isinstance(ranges, dict) and 'min' in ranges:
                    simulated_values = np.random.triangular(
                        ranges['min'],
                        ranges['most_likely'],
                        ranges['max'],
                        size=n_simulations
                    )
                    category_results[item] = simulated_values
            if category_results:
                results[category] = category_results
        return results

    def calculate_npv(self, cash_flows: np.ndarray, discount_rate: float) -> float:
        periods = np.arange(len(cash_flows))
        return np.sum(cash_flows / (1 + discount_rate) ** periods)

    def generate_time_series(self, base_value: float, periods: int,
                             trend: float = 0, seasonality: float = 0,
                             noise: float = 0) -> np.ndarray:
        time = np.arange(periods)
        trend_component = trend * time
        seasonal_component = seasonality * np.sin(2 * np.pi * time / 12)
        noise_component = noise * np.random.randn(periods)
        return base_value + trend_component + seasonal_component + noise_component

    def calculate_sensitivity(self, params: Dict, target_func,
                              delta: float = 0.1) -> pd.DataFrame:
        base_case = target_func(params)
        sensitivities = []

        def modify_params(params: Dict, param_path: List[str],
                          factor: float) -> Dict:
            modified = params.copy()
            current = modified
            for key in param_path[:-1]:
                current = current[key]
            current[param_path[-1]] *= (1 + factor)
            return modified

        def get_nested_params(d: Dict, prefix: List[str] = None) -> List[List[str]]:
            if prefix is None:
                prefix = []
            result = []
            for k, v in d.items():
                if isinstance(v, dict):
                    if 'min' in v:
                        result.append(prefix + [k])
                    else:
                        result.extend(get_nested_params(v, prefix + [k]))
            return result

        param_paths = get_nested_params(params)

        for path in param_paths:
            high_case = target_func(modify_params(params, path, delta))
            low_case = target_func(modify_params(params, path, -delta))

            sensitivity = {
                'Parameter': '_'.join(path),
                'Low_Case': low_case,
                'Base_Case': base_case,
                'High_Case': high_case,
                'Swing': high_case - low_case,
                'Normalized_Sensitivity': (high_case - low_case) / (2 * delta * base_case)
            }
            sensitivities.append(sensitivity)

        return pd.DataFrame(sensitivities)


class WellDataManager:
    def __init__(self, core: WellAnalysisCore):
        self.core = core
        self.wells_data = {}
        self.production_history = {}
        self.maintenance_history = {}
        self.environmental_history = {}
        self.financial_history = {}

    def load_data(self, file_paths: Dict[str, str]) -> None:
        try:
            for key, path in file_paths.items():
                if key == 'wells':
                    self.wells_data = pd.read_excel(path, sheet_name=None)
                elif key == 'production':
                    self.production_history = pd.read_excel(path, sheet_name=None)
                elif key == 'maintenance':
                    self.maintenance_history = pd.read_excel(path, sheet_name=None)
                elif key == 'environmental':
                    self.environmental_history = pd.read_excel(path, sheet_name=None)
                elif key == 'financial':
                    self.financial_history = pd.read_excel(path, sheet_name=None)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def get_well_data(self, well_name: str) -> Dict[str, pd.DataFrame]:
        return {
            'production': self.production_history.get(well_name),
            'maintenance': self.maintenance_history.get(well_name),
            'environmental': self.environmental_history.get(well_name),
            'financial': self.financial_history.get(well_name)
        }

    def calculate_well_metrics(self, well_name: str) -> Dict[str, float]:
        well_data = self.get_well_data(well_name)
        if not well_data['production'] is None:
            prod_data = well_data['production']
            return {
                'avg_oil_rate': prod_data['Oil_Production_BBL'].mean(),
                'avg_gas_rate': prod_data['Gas_Production_MCF'].mean(),
                'avg_water_cut': prod_data['Water_Cut_Percentage'].mean(),
                'cumulative_oil': prod_data['Oil_Production_BBL'].sum(),
                'cumulative_gas': prod_data['Gas_Production_MCF'].sum(),
                'days_produced': len(prod_data)
            }
        return {}

    def export_results(self, results: Dict, filename: str) -> None:
        try:
            with pd.ExcelWriter(filename) as writer:
                for sheet_name, data in results.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name)
                    elif isinstance(data, dict):
                        pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name)
                    else:
                        pd.DataFrame([data]).to_excel(writer, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Error exporting results: {str(e)}")

class WellVisualizationEngine:
    def __init__(self, core: WellAnalysisCore):
        self.core = core
        self.color_scale = px.colors.sequential.Blues
        self.theme = {
            'background': '#ffffff',
            'text': '#000000',
            'grid': '#e5e5e5',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }

    def create_parameter_hierarchy_tree(self, params: dict, width: int = 1000, height: int = 800) -> go.Figure:
        G = nx.Graph()
        edges = []
        node_values = {}

        def add_nodes(d, parent=None, prefix=""):
            for key, value in d.items():
                node_id = f"{prefix}{key}"
                if isinstance(value, dict):
                    if 'min' in value:
                        G.add_node(node_id, value=value['most_likely'])
                        node_values[node_id] = value['most_likely']
                    else:
                        G.add_node(node_id, value=0)
                        add_nodes(value, node_id, f"{node_id}_")
                else:
                    G.add_node(node_id, value=value)
                    node_values[node_id] = value
                if parent:
                    edges.append((parent, node_id))

        add_nodes(params)
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Parameter Value',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
            node_trace['marker']['color'] += tuple([node_values.get(node, 0)])

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Parameter Hierarchy',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            width=width,
                            height=height,
                            annotations=[dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def create_sensitivity_tornado(self, sensitivity_data: pd.DataFrame,
                                   width: int = 800, height: int = 600) -> go.Figure:
        sensitivity_data = sensitivity_data.sort_values('Swing', ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=sensitivity_data['Parameter'],
            x=sensitivity_data['High_Case'] - sensitivity_data['Base_Case'],
            orientation='h',
            name='Positive Impact',
            marker_color='green',
            showlegend=True
        ))

        fig.add_trace(go.Bar(
            y=sensitivity_data['Parameter'],
            x=sensitivity_data['Low_Case'] - sensitivity_data['Base_Case'],
            orientation='h',
            name='Negative Impact',
            marker_color='red',
            showlegend=True
        ))

        fig.update_layout(
            title='Sensitivity Analysis - Tornado Chart',
            barmode='overlay',
            width=width,
            height=height,
            yaxis={'categoryorder': 'array', 'categoryarray': sensitivity_data['Parameter']},
            xaxis_title='Impact on Base Case',
            showlegend=True
        )

        return fig

    def create_monte_carlo_results(self, results: Dict[str, np.ndarray],
                                   width: int = 800, height: int = 600) -> go.Figure:
        fig = make_subplots(rows=len(results), cols=1,
                            subplot_titles=list(results.keys()))

        row = 1
        for category, values in results.items():
            fig.add_trace(
                go.Histogram(x=values, name=category, nbinsx=30),
                row=row, col=1
            )
            row += 1

        fig.update_layout(
            title='Monte Carlo Simulation Results',
            height=height * len(results),
            width=width,
            showlegend=True
        )

        return fig

    def create_time_series_plot(self, df: pd.DataFrame,
                                columns: List[str],
                                title: str = "Time Series Plot",
                                width: int = 800,
                                height: int = 400) -> go.Figure:
        fig = go.Figure()

        for col in columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col)
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            width=width,
            height=height,
            showlegend=True
        )

        return fig

    def create_cost_breakdown_sunburst(self, data: Dict,
                                       width: int = 800,
                                       height: int = 800) -> go.Figure:
        # Prepare data for sunburst chart
        ids = []
        labels = []
        parents = []
        values = []

        def process_dict(d, parent=""):
            for key, value in d.items():
                current_id = f"{parent}_{key}" if parent else key
                ids.append(current_id)
                labels.append(key)
                parents.append(parent)

                if isinstance(value, dict):
                    if 'min' in value:
                        values.append(value['most_likely'])
                    else:
                        values.append(sum(
                            [v['most_likely'] if isinstance(v, dict) and 'min' in v
                             else (sum(v.values()) if isinstance(v, dict) else v)
                             for v in value.values()]
                        ))
                        process_dict(value, current_id)
                else:
                    values.append(value)

        process_dict(data)

        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        ))

        fig.update_layout(
            width=width,
            height=height,
            title="Cost Breakdown Structure"
        )

        return fig

    def create_correlation_heatmap(self, df: pd.DataFrame,
                                   width: int = 800,
                                   height: int = 800) -> go.Figure:
        correlation_matrix = df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
        ))

        fig.update_layout(
            title='Parameter Correlation Heatmap',
            width=width,
            height=height,
            xaxis_title="Parameters",
            yaxis_title="Parameters"
        )

        return fig

    def create_parameter_distribution(self, param_name: str,
                                      param_config: Dict[str, float],
                                      width: int = 600,
                                      height: int = 400) -> go.Figure:
        x = np.linspace(param_config['min'], param_config['max'], 1000)
        y = stats.triang.pdf(
            x,
            c=(param_config['most_likely'] - param_config['min']) /
              (param_config['max'] - param_config['min']),
            loc=param_config['min'],
            scale=param_config['max'] - param_config['min']
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            fill='tozeroy',
            name=param_name
        ))

        fig.add_vline(
            x=param_config['most_likely'],
            line_dash="dash",
            annotation_text="Most Likely"
        )

        fig.update_layout(
            title=f'{param_name} Distribution',
            xaxis_title="Value",
            yaxis_title="Probability Density",
            width=width,
            height=height,
            showlegend=True
        )

        return fig