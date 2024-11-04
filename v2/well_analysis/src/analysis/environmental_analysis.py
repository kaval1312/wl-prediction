import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from ..core.data_structures import WellData
from ..core.constants import ENVIRONMENTAL_CATEGORIES

logger = logging.getLogger(__name__)


class EnvironmentalAnalyzer:
    def __init__(self):
        """Initialize environmental analyzer"""
        self.results = {}

    def analyze_emissions(self, well_data: WellData) -> Dict:
        """Analyze emissions data"""
        try:
            env_data = well_data.environmental_data

            # Get all emission-related columns
            emission_cols = [col for col in env_data.columns
                             if 'Emissions' in col]

            # Calculate emission metrics
            emission_metrics = {}

            # Total emissions
            for col in emission_cols:
                emission_metrics[f'total_{col}'] = env_data[col].sum()
                emission_metrics[f'average_{col}'] = env_data[col].mean()
                emission_metrics[f'max_{col}'] = env_data[col].max()
                emission_metrics[f'min_{col}'] = env_data[col].min()
                emission_metrics[f'std_{col}'] = env_data[col].std()

                # Calculate trends (using simple linear regression)
                if len(env_data) > 1:
                    x = np.arange(len(env_data))
                    y = env_data[col].values
                    trend = np.polyfit(x, y, 1)[0]
                    emission_metrics[f'{col}_trend'] = trend

            # Normalize by production
            total_production = well_data.production_data['Oil_Production_BBL'].sum()
            if total_production > 0:
                for col in emission_cols:
                    emission_metrics[f'{col}_per_bbl'] = (
                            env_data[col].sum() / total_production
                    )

            return emission_metrics

        except Exception as e:
            logger.error(f"Error analyzing emissions: {str(e)}")
            return {}

    def analyze_water_management(self, well_data: WellData) -> Dict:
        """Analyze water management data"""
        try:
            env_data = well_data.environmental_data
            prod_data = well_data.production_data

            water_metrics = {}

            # Produced water analysis
            water_metrics['total_produced_water'] = prod_data['Water_Cut_Percentage'].sum()
            water_metrics['avg_water_cut'] = prod_data['Water_Cut_Percentage'].mean()
            water_metrics['max_water_cut'] = prod_data['Water_Cut_Percentage'].max()

            # Water management costs
            water_cols = [col for col in env_data.columns if 'Water_Management' in col]
            for col in water_cols:
                water_metrics[f'total_{col}_cost'] = env_data[col].sum()
                water_metrics[f'avg_{col}_cost'] = env_data[col].mean()
                water_metrics[f'max_{col}_cost'] = env_data[col].max()

            # Water recycling metrics
            if 'Water_Management_Produced_Water_Recycling' in env_data.columns:
                recycling = env_data['Water_Management_Produced_Water_Recycling']
                water_metrics['total_recycled_water'] = recycling.sum()
                water_metrics['recycling_rate'] = (
                    recycling.sum() / water_metrics['total_produced_water']
                    if water_metrics['total_produced_water'] > 0 else 0
                )

            # Calculate trends
            if len(prod_data) > 1:
                x = np.arange(len(prod_data))
                y = prod_data['Water_Cut_Percentage'].values
                water_cut_trend = np.polyfit(x, y, 1)[0]
                water_metrics['water_cut_trend'] = water_cut_trend

            return water_metrics

        except Exception as e:
            logger.error(f"Error analyzing water management: {str(e)}")
            return {}

    def analyze_soil_management(self, well_data: WellData) -> Dict:
        """Analyze soil management and contamination data"""
        try:
            env_data = well_data.environmental_data

            soil_metrics = {}

            # Get soil-related columns
            soil_cols = [col for col in env_data.columns if 'Soil_Management' in col]

            for col in soil_cols:
                soil_metrics[f'total_{col}_cost'] = env_data[col].sum()
                soil_metrics[f'avg_{col}_cost'] = env_data[col].mean()
                soil_metrics[f'max_{col}_cost'] = env_data[col].max()

            # Analyze contamination monitoring
            if 'Soil_Management_Contamination_Monitoring_Sampling' in env_data.columns:
                sampling = env_data['Soil_Management_Contamination_Monitoring_Sampling']
                soil_metrics['sampling_frequency'] = len(sampling.dropna()) / len(sampling)
                soil_metrics['total_sampling_cost'] = sampling.sum()

            # Analyze remediation costs
            remediation_cols = [col for col in soil_cols if 'Remediation' in col]
            if remediation_cols:
                total_remediation = env_data[remediation_cols].sum().sum()
                soil_metrics['total_remediation_cost'] = total_remediation

            return soil_metrics

        except Exception as e:
            logger.error(f"Error analyzing soil management: {str(e)}")
            return {}

    def analyze_waste_management(self, well_data: WellData) -> Dict:
        """Analyze waste management data"""
        try:
            env_data = well_data.environmental_data

            waste_metrics = {}

            # Get waste-related columns
            waste_cols = [col for col in env_data.columns if 'Waste_Management' in col]

            # Calculate waste management costs
            for col in waste_cols:
                waste_metrics[f'total_{col}_cost'] = env_data[col].sum()
                waste_metrics[f'avg_{col}_cost'] = env_data[col].mean()

            # Calculate waste metrics by type
            waste_types = ['Drilling_Waste', 'Chemical_Waste', 'General_Waste']
            for waste_type in waste_types:
                type_cols = [col for col in waste_cols if waste_type in col]
                if type_cols:
                    total_type_cost = env_data[type_cols].sum().sum()
                    waste_metrics[f'total_{waste_type}_cost'] = total_type_cost

            # Calculate cost per barrel of oil
            total_oil = well_data.production_data['Oil_Production_BBL'].sum()
            if total_oil > 0:
                total_waste_cost = sum(waste_metrics[k] for k in waste_metrics
                                       if k.startswith('total_') and k.endswith('_cost'))
                waste_metrics['waste_cost_per_bbl'] = total_waste_cost / total_oil

            return waste_metrics

        except Exception as e:
            logger.error(f"Error analyzing waste management: {str(e)}")
            return {}

    def analyze_wildlife_protection(self, well_data: WellData) -> Dict:
        """Analyze wildlife protection measures"""
        try:
            env_data = well_data.environmental_data

            wildlife_metrics = {}

            # Get wildlife-related columns
            wildlife_cols = [col for col in env_data.columns
                             if 'Wildlife_Protection' in col]

            # Calculate protection costs
            for col in wildlife_cols:
                wildlife_metrics[f'total_{col}_cost'] = env_data[col].sum()
                wildlife_metrics[f'avg_{col}_cost'] = env_data[col].mean()

            # Analyze habitat preservation
            habitat_cols = [col for col in wildlife_cols if 'Habitat' in col]
            if habitat_cols:
                total_habitat_cost = env_data[habitat_cols].sum().sum()
                wildlife_metrics['total_habitat_cost'] = total_habitat_cost

            # Analyze species monitoring
            monitoring_cols = [col for col in wildlife_cols if 'Monitoring' in col]
            if monitoring_cols:
                total_monitoring_cost = env_data[monitoring_cols].sum().sum()
                wildlife_metrics['total_monitoring_cost'] = total_monitoring_cost

            return wildlife_metrics

        except Exception as e:
            logger.error(f"Error analyzing wildlife protection: {str(e)}")
            return {}

    def calculate_environmental_costs(self, well_data: WellData) -> Dict:
        """Calculate total environmental costs and metrics"""
        try:
            env_data = well_data.environmental_data

            # Initialize cost categories
            total_costs = {category: 0 for category in ENVIRONMENTAL_CATEGORIES.keys()}

            # Calculate costs for each category
            for category in ENVIRONMENTAL_CATEGORIES.keys():
                category_cols = [col for col in env_data.columns
                                 if col.startswith(category)]
                total_costs[category] = env_data[category_cols].sum().sum()

            # Calculate total environmental cost
            total_env_cost = sum(total_costs.values())

            # Calculate cost per barrel
            total_oil = well_data.production_data['Oil_Production_BBL'].sum()
            cost_per_barrel = total_env_cost / total_oil if total_oil > 0 else 0

            # Calculate cost percentages
            cost_percentages = {
                category: (cost / total_env_cost * 100 if total_env_cost > 0 else 0)
                for category, cost in total_costs.items()
            }

            return {
                'total_environmental_cost': total_env_cost,
                'cost_per_barrel': cost_per_barrel,
                'category_costs': total_costs,
                'category_percentages': cost_percentages
            }

        except Exception as e:
            logger.error(f"Error calculating environmental costs: {str(e)}")
            return {}

    def calculate_environmental_metrics(self, well_data: WellData) -> Dict:
        """Calculate comprehensive environmental metrics"""
        try:
            # Combine all environmental analyses
            metrics = {}

            # Get emissions metrics
            metrics['emissions'] = self.analyze_emissions(well_data)

            # Get water management metrics
            metrics['water'] = self.analyze_water_management(well_data)

            # Get soil management metrics
            metrics['soil'] = self.analyze_soil_management(well_data)

            # Get waste management metrics
            metrics['waste'] = self.analyze_waste_management(well_data)

            # Get wildlife protection metrics
            metrics['wildlife'] = self.analyze_wildlife_protection(well_data)

            # Get total environmental costs
            metrics['costs'] = self.calculate_environmental_costs(well_data)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating environmental metrics: {str(e)}")
            return {}

    def generate_environmental_report(self, well_data: WellData) -> Dict:
        """Generate comprehensive environmental report"""
        try:
            report = {}

            # Get all metrics
            metrics = self.calculate_environmental_metrics(well_data)

            # Calculate environmental KPIs
            kpis = {
                'total_environmental_cost': metrics['costs']['total_environmental_cost'],
                'env_cost_per_barrel': metrics['costs']['cost_per_barrel'],
                'total_emissions': sum(v for k, v in metrics['emissions'].items()
                                       if k.startswith('total_')),
                'water_recycling_rate': metrics['water'].get('recycling_rate', 0),
                'waste_management_efficiency': metrics['waste'].get('waste_cost_per_bbl', 0)
            }

            # Add trends
            trends = {
                'emission_trend': metrics['emissions'].get('Methane_Emissions_trend', 0),
                'water_cut_trend': metrics['water'].get('water_cut_trend', 0)
            }

            # Add recommendations based on metrics
            recommendations = []
            if kpis['env_cost_per_barrel'] > 5:  # Example threshold
                recommendations.append("Consider cost reduction measures")
            if kpis['water_recycling_rate'] < 0.5:  # Example threshold
                recommendations.append("Improve water recycling program")

            report = {
                'kpis': kpis,
                'trends': trends,
                'detailed_metrics': metrics,
                'recommendations': recommendations
            }

            return report

        except Exception as e:
            logger.error(f"Error generating environmental report: {str(e)}")
            return {}