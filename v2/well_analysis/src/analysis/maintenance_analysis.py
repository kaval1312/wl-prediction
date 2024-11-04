import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from ..core.data_structures import WellData
from ..core.constants import MAINTENANCE_ITEMS

logger = logging.getLogger(__name__)


class MaintenanceAnalyzer:
    def __init__(self):
        """Initialize maintenance analyzer"""
        self.results = {}

    def analyze_maintenance_events(self, well_data: WellData) -> Dict:
        """Analyze maintenance events and their statistics"""
        try:
            maint_data = well_data.maintenance_data

            # Group by maintenance category
            category_stats = {}
            for category in MAINTENANCE_ITEMS.keys():
                category_data = maint_data[maint_data['Maintenance_Category'] == category]

                if not category_data.empty:
                    category_stats[category] = {
                        'total_cost': category_data['Cost'].sum(),
                        'avg_cost': category_data['Cost'].mean(),
                        'num_events': len(category_data),
                        'min_cost': category_data['Cost'].min(),
                        'max_cost': category_data['Cost'].max(),
                        'std_cost': category_data['Cost'].std(),
                        'last_event': category_data['Date'].max(),
                        'first_event': category_data['Date'].min()
                    }

                    # Calculate frequency
                    date_range = (category_data['Date'].max() -
                                  category_data['Date'].min()).days
                    if date_range > 0:
                        category_stats[category]['avg_frequency_days'] = (
                                date_range / len(category_data)
                        )

                    # Priority distribution
                    priority_dist = category_data['Priority'].value_counts()
                    category_stats[category]['priority_distribution'] = (
                        priority_dist.to_dict()
                    )

            return category_stats

        except Exception as e:
            logger.error(f"Error analyzing maintenance events: {str(e)}")
            return {}

    def analyze_cost_trends(self, well_data: WellData,
                            window: int = 30) -> Dict[str, pd.Series]:
        """Analyze maintenance cost trends"""
        try:
            maint_data = well_data.maintenance_data.copy()
            maint_data = maint_data.sort_values('Date')

            # Create daily cost aggregation
            daily_costs = maint_data.groupby('Date')['Cost'].sum()

            # Calculate rolling statistics
            trends = {
                'rolling_mean': daily_costs.rolling(window=window).mean(),
                'rolling_std': daily_costs.rolling(window=window).std(),
                'cumulative_cost': daily_costs.cumsum(),
                'daily_cost': daily_costs
            }

            # Calculate trend line
            x = np.arange(len(daily_costs))
            if len(x) > 1:
                z = np.polyfit(x, daily_costs.values, 1)
                trends['trend_line'] = np.poly1d(z)(x)

            return trends

        except Exception as e:
            logger.error(f"Error analyzing cost trends: {str(e)}")
            return {}

    def analyze_equipment_reliability(self, well_data: WellData) -> Dict:
        """Analyze equipment reliability and failure patterns"""
        try:
            maint_data = well_data.maintenance_data

            reliability_metrics = {}

            # Analyze each equipment type
            equipment_failures = maint_data[maint_data['Maintenance_Category'] ==
                                            'Emergency_Repairs']

            if not equipment_failures.empty:
                for item in equipment_failures['Maintenance_Item'].unique():
                    item_failures = equipment_failures[
                        equipment_failures['Maintenance_Item'] == item]

                    # Calculate metrics
                    total_failures = len(item_failures)
                    total_cost = item_failures['Cost'].sum()
                    avg_repair_cost = item_failures['Cost'].mean()

                    # Calculate time between failures
                    if total_failures > 1:
                        dates = sorted(item_failures['Date'])
                        mtbf = np.mean([(dates[i + 1] - dates[i]).days
                                        for i in range(len(dates) - 1)])
                    else:
                        mtbf = None

                    reliability_metrics[item] = {
                        'total_failures': total_failures,
                        'total_cost': total_cost,
                        'avg_repair_cost': avg_repair_cost,
                        'mtbf_days': mtbf
                    }

            return reliability_metrics

        except Exception as e:
            logger.error(f"Error analyzing equipment reliability: {str(e)}")
            return {}

    def predict_maintenance_needs(self, well_data: WellData,
                                  forecast_days: int = 90) -> Dict:
        """Predict future maintenance needs based on historical patterns"""
        try:
            maint_data = well_data.maintenance_data

            predictions = {}

            # Analyze each maintenance category
            for category in MAINTENANCE_ITEMS.keys():
                category_data = maint_data[
                    maint_data['Maintenance_Category'] == category]

                if not category_data.empty:
                    # Calculate average frequency
                    date_range = (category_data['Date'].max() -
                                  category_data['Date'].min()).days
                    if date_range > 0:
                        frequency = date_range / len(category_data)

                        # Predict next event
                        last_event = category_data['Date'].max()
                        next_event = last_event + timedelta(days=frequency)

                        # Predict number of events in forecast period
                        num_events = forecast_days / frequency

                        # Predict costs
                        avg_cost = category_data['Cost'].mean()
                        predicted_cost = avg_cost * num_events

                        predictions[category] = {
                            'next_predicted_event': next_event,
                            'predicted_num_events': num_events,
                            'predicted_total_cost': predicted_cost,
                            'confidence_level': 'Medium'  # Placeholder for more sophisticated prediction
                        }

            return predictions

        except Exception as e:
            logger.error(f"Error predicting maintenance needs: {str(e)}")
            return {}

    def optimize_maintenance_schedule(self, well_data: WellData,
                                      schedule_days: int = 90) -> Dict:
        """Optimize maintenance schedule based on historical patterns"""
        try:
            maint_data = well_data.maintenance_data

            # Get predicted maintenance needs
            predictions = self.predict_maintenance_needs(well_data, schedule_days)

            schedule = []
            total_cost = 0

            # Create schedule entries
            current_date = datetime.now()
            end_date = current_date + timedelta(days=schedule_days)

            for category, pred in predictions.items():
                if 'next_predicted_event' in pred:
                    next_event = pred['next_predicted_event']
                    while next_event <= end_date:
                        schedule.append({
                            'date': next_event,
                            'category': category,
                            'estimated_cost': pred['predicted_total_cost'] /
                                              pred['predicted_num_events'],
                            'priority': 'Medium',
                            'type': 'Predicted'
                        })
                        # Add frequency to get next event
                        next_event += timedelta(
                            days=schedule_days / pred['predicted_num_events']
                        )
                        total_cost += pred['predicted_total_cost'] / pred['predicted_num_events']

            # Sort schedule by date
            schedule = sorted(schedule, key=lambda x: x['date'])

            return {
                'schedule': schedule,
                'total_cost': total_cost,
                'num_events': len(schedule)
            }

        except Exception as e:
            logger.error(f"Error optimizing maintenance schedule: {str(e)}")
            return {}

    def calculate_maintenance_kpis(self, well_data: WellData) -> Dict:
        """Calculate key performance indicators for maintenance"""
        try:
            maint_data = well_data.maintenance_data

            kpis = {}

            # Calculate overall maintenance costs
            total_cost = maint_data['Cost'].sum()
            avg_monthly_cost = (maint_data.groupby(
                pd.Grouper(key='Date', freq='M'))['Cost'].sum().mean())

            # Calculate maintenance event frequencies
            total_events = len(maint_data)
            unique_dates = maint_data['Date'].nunique()
            avg_events_per_day = total_events / unique_dates if unique_dates > 0 else 0

            # Calculate emergency vs planned maintenance ratio
            emergency_events = len(maint_data[
                                       maint_data['Maintenance_Category'] == 'Emergency_Repairs'])
            planned_events = total_events - emergency_events
            emergency_ratio = (emergency_events / total_events
                               if total_events > 0 else 0)

            # Calculate MTBF for emergency repairs
            emergency_repairs = maint_data[
                maint_data['Maintenance_Category'] == 'Emergency_Repairs']
            if len(emergency_repairs) > 1:
                dates = sorted(emergency_repairs['Date'])
                mtbf = np.mean([(dates[i + 1] - dates[i]).days
                                for i in range(len(dates) - 1)])
            else:
                mtbf = None

            # Calculate cost per BOE
            total_oil = well_data.production_data['Oil_Production_BBL'].sum()
            total_gas = well_data.production_data['Gas_Production_MCF'].sum()
            total_boe = total_oil + total_gas / 6
            cost_per_boe = total_cost / total_boe if total_boe > 0 else 0

            kpis = {
                'total_maintenance_cost': total_cost,
                'average_monthly_cost': avg_monthly_cost,
                'total_maintenance_events': total_events,
                'avg_events_per_day': avg_events_per_day,
                'emergency_ratio': emergency_ratio,
                'mtbf_days': mtbf,
                'cost_per_boe': cost_per_boe,
                'planned_maintenance_ratio': 1 - emergency_ratio
            }

            return kpis

        except Exception as e:
            logger.error(f"Error calculating maintenance KPIs: {str(e)}")
            return {}

    def analyze_maintenance_impact(self, well_data: WellData) -> Dict:
        """Analyze impact of maintenance on production"""
        try:
            maint_data = well_data.maintenance_data
            prod_data = well_data.production_data

            impact_analysis = {}

            # Analyze production changes around maintenance events
            for category in MAINTENANCE_ITEMS.keys():
                category_events = maint_data[
                    maint_data['Maintenance_Category'] == category]

                if not category_events.empty:
                    production_impacts = []

                    for _, event in category_events.iterrows():
                        event_date = event['Date']

                        # Get production before and after event
                        pre_production = prod_data[
                            prod_data['Date'].between(
                                event_date - timedelta(days=7),
                                event_date
                            )]['Oil_Production_BBL'].mean()

                        post_production = prod_data[
                            prod_data['Date'].between(
                                event_date,
                                event_date + timedelta(days=7)
                            )]['Oil_Production_BBL'].mean()

                        if pre_production > 0:
                            impact = (post_production - pre_production) / pre_production
                            production_impacts.append(impact)

                    if production_impacts:
                        impact_analysis[category] = {
                            'avg_production_impact': np.mean(production_impacts),
                            'max_production_impact': np.max(production_impacts),
                            'min_production_impact': np.min(production_impacts),
                            'std_production_impact': np.std(production_impacts)
                        }

            return impact_analysis

        except Exception as e:
            logger.error(f"Error analyzing maintenance impact: {str(e)}")
            return {}

    def generate_maintenance_report(self, well_data: WellData) -> Dict:
        """Generate comprehensive maintenance report"""
        try:
            report = {}

            # Get all maintenance analyses
            report['events'] = self.analyze_maintenance_events(well_data)
            report['cost_trends'] = self.analyze_cost_trends(well_data)
            report['reliability'] = self.analyze_equipment_reliability(well_data)
            report['predictions'] = self.predict_maintenance_needs(well_data)
            report['schedule'] = self.optimize_maintenance_schedule(well_data)
            report['kpis'] = self.calculate_maintenance_kpis(well_data)
            report['impact'] = self.analyze_maintenance_impact(well_data)

            # Generate recommendations
            recommendations = []

            # Check KPIs for issues
            if report['kpis'].get('emergency_ratio', 0) > 0.3:
                recommendations.append(
                    "High emergency maintenance ratio - consider increasing preventive maintenance"
                )

            if report['kpis'].get('cost_per_boe', 0) > 5:  # Example threshold
                recommendations.append(
                    "High maintenance cost per BOE - review maintenance procedures"
                )

            # Add recommendations based on reliability
            for equipment, metrics in report['reliability'].items():
                if metrics.get('mtbf_days', float('inf')) < 30:  # Example threshold
                    recommendations.append(
                        f"Frequent failures for {equipment} - consider replacement"
                    )

            report['recommendations'] = recommendations

            return report

        except Exception as e:
            logger.error(f"Error generating maintenance report: {str(e)}")
            return {}