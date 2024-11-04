import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from ..core.data_structures import WellData
from ..core.constants import MONTE_CARLO_PARAMS

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    def __init__(self, seed: int = 42):
        """Initialize Monte Carlo simulator"""
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def run_production_simulation(self, well_data: WellData,
                                  n_simulations: int = 1000,
                                  forecast_days: int = 365) -> Dict:
        """Run Monte Carlo simulation for production forecasting"""
        try:
            results = {
                'oil_production': np.zeros((n_simulations, forecast_days)),
                'gas_production': np.zeros((n_simulations, forecast_days)),
                'water_cut': np.zeros((n_simulations, forecast_days))
            }

            prod_params = MONTE_CARLO_PARAMS['Production']

            for sim in range(n_simulations):
                # Generate random factors
                oil_factor = np.random.triangular(
                    prod_params['Oil_Rate']['min'],
                    prod_params['Oil_Rate']['most_likely'],
                    prod_params['Oil_Rate']['max']
                )
                gas_factor = np.random.triangular(
                    prod_params['Gas_Rate']['min'],
                    prod_params['Gas_Rate']['most_likely'],
                    prod_params['Gas_Rate']['max']
                )
                water_factor = np.random.triangular(
                    prod_params['Water_Cut']['min'],
                    prod_params['Water_Cut']['most_likely'],
                    prod_params['Water_Cut']['max']
                )

                # Generate decline curves
                for t in range(forecast_days):
                    decline = 1 / (1 + 0.1 * t / 365)  # Example decline function
                    results['oil_production'][sim, t] = (
                            well_data.production_data['Oil_Production_BBL'].iloc[-1] *
                            oil_factor * decline
                    )
                    results['gas_production'][sim, t] = (
                            well_data.production_data['Gas_Production_MCF'].iloc[-1] *
                            gas_factor * decline
                    )
                    results['water_cut'][sim, t] = (
                            well_data.production_data['Water_Cut_Percentage'].iloc[-1] *
                            water_factor
                    )

            return results

        except Exception as e:
            logger.error(f"Error in production simulation: {str(e)}")
            return {}

    def run_economic_simulation(self, well_data: WellData,
                                n_simulations: int = 1000,
                                forecast_days: int = 365) -> Dict:
        """Run Monte Carlo simulation for economic forecasting"""
        try:
            results = {
                'npv': np.zeros(n_simulations),
                'revenue': np.zeros((n_simulations, forecast_days)),
                'opex': np.zeros((n_simulations, forecast_days)),
                'cash_flow': np.zeros((n_simulations, forecast_days))
            }

            # Get price and cost parameters
            price_params = MONTE_CARLO_PARAMS['Prices']
            cost_params = MONTE_CARLO_PARAMS['Costs']

            # Run production simulation first
            prod_results = self.run_production_simulation(
                well_data, n_simulations, forecast_days)

            for sim in range(n_simulations):
                # Generate random factors
                oil_price = np.random.triangular(
                    price_params['Oil_Price']['min'],
                    price_params['Oil_Price']['most_likely'],
                    price_params['Oil_Price']['max']
                )
                gas_price = np.random.triangular(
                    price_params['Gas_Price']['min'],
                    price_params['Gas_Price']['most_likely'],
                    price_params['Gas_Price']['max']
                )
                opex_factor = np.random.triangular(
                    cost_params['Operating_Cost']['min'],
                    cost_params['Operating_Cost']['most_likely'],
                    cost_params['Operating_Cost']['max']
                )

                # Calculate daily revenues and costs
                for t in range(forecast_days):
                    results['revenue'][sim, t] = (
                            prod_results['oil_production'][sim, t] * oil_price +
                            prod_results['gas_production'][sim, t] * gas_price
                    )
                    results['opex'][sim, t] = (
                            well_data.financial_data['Labor_Operations_Total'].mean() *
                            opex_factor
                    )
                    results['cash_flow'][sim, t] = (
                            results['revenue'][sim, t] - results['opex'][sim, t]
                    )

                # Calculate NPV
                discount_rate = 0.1  # Example discount rate
                discount_factors = 1 / (1 + discount_rate) ** np.arange(forecast_days)
                results['npv'][sim] = np.sum(
                    results['cash_flow'][sim, :] * discount_factors
                )

            return results

        except Exception as e:
            logger.error(f"Error in economic simulation: {str(e)}")
            return {}

    def run_maintenance_simulation(self, well_data: WellData,
                                   n_simulations: int = 1000,
                                   forecast_days: int = 365) -> Dict:
        """Run Monte Carlo simulation for maintenance forecasting"""
        try:
            results = {
                'maintenance_cost': np.zeros((n_simulations, forecast_days)),
                'num_events': np.zeros((n_simulations, forecast_days)),
                'total_cost': np.zeros(n_simulations)
            }

            maint_params = MONTE_CARLO_PARAMS['Costs']['Maintenance_Cost']

            for sim in range(n_simulations):
                # Generate random factor
                maint_factor = np.random.triangular(
                    maint_params['min'],
                    maint_params['most_likely'],
                    maint_params['max']
                )

                # Calculate historical event probability
                historical_events = len(well_data.maintenance_data)
                historical_days = len(well_data.production_data)
                daily_prob = historical_events / historical_days

                # Simulate maintenance events
                for t in range(forecast_days):
                    # Simulate event occurrence
                    if np.random.random() < daily_prob:
                        results['num_events'][sim, t] = 1
                        # Generate random cost based on historical data
                        base_cost = well_data.maintenance_data['Cost'].mean()
                        results['maintenance_cost'][sim, t] = base_cost * maint_factor

                results['total_cost'][sim] = results['maintenance_cost'][sim, :].sum()

            return results

        except Exception as e:
            logger.error(f"Error in maintenance simulation: {str(e)}")
            return {}

    def run_environmental_simulation(self, well_data: WellData,
                                     n_simulations: int = 1000,
                                     forecast_days: int = 365) -> Dict:
        """Run Monte Carlo simulation for environmental metrics"""
        try:
            results = {
                'emissions': np.zeros((n_simulations, forecast_days)),
                'water_treatment': np.zeros((n_simulations, forecast_days)),
                'total_env_cost': np.zeros(n_simulations)
            }

            env_params = MONTE_CARLO_PARAMS['Environmental']

            for sim in range(n_simulations):
                # Generate random factors
                emission_factor = np.random.triangular(
                    env_params['Emission_Rates']['min'],
                    env_params['Emission_Rates']['most_likely'],
                    env_params['Emission_Rates']['max']
                )
                water_factor = np.random.triangular(
                    env_params['Water_Treatment_Cost']['min'],
                    env_params['Water_Treatment_Cost']['most_likely'],
                    env_params['Water_Treatment_Cost']['max']
                )

                # Calculate daily environmental metrics
                for t in range(forecast_days):
                    # Base values from historical data
                    base_emissions = well_data.environmental_data[
                        [col for col in well_data.environmental_data.columns
                         if 'Emissions' in col]
                    ].mean().mean()
                    base_water = well_data.environmental_data[
                        [col for col in well_data.environmental_data.columns
                         if 'Water' in col]
                    ].mean().mean()

                    results['emissions'][sim, t] = base_emissions * emission_factor
                    results['water_treatment'][sim, t] = base_water * water_factor

                results['total_env_cost'][sim] = (
                        results['emissions'][sim, :].sum() +
                        results['water_treatment'][sim, :].sum()
                )

            return results

        except Exception as e:
            logger.error(f"Error in environmental simulation: {str(e)}")
            return {}

    def run_integrated_simulation(self, well_data: WellData,
                                  n_simulations: int = 1000,
                                  forecast_days: int = 365) -> Dict:
        """Run integrated Monte Carlo simulation combining all aspects"""
        try:
            # Run individual simulations
            prod_results = self.run_production_simulation(
                well_data, n_simulations, forecast_days)
            econ_results = self.run_economic_simulation(
                well_data, n_simulations, forecast_days)
            maint_results = self.run_maintenance_simulation(
                well_data, n_simulations, forecast_days)
            env_results = self.run_environmental_simulation(
                well_data, n_simulations, forecast_days)

            # Combine results
            integrated_results = {
                'production': prod_results,
                'economics': econ_results,
                'maintenance': maint_results,
                'environmental': env_results,
                'integrated_metrics': {}
            }

            # Calculate integrated metrics
            for sim in range(n_simulations):
                # Total cost including all aspects
                total_cost = (
                        econ_results['opex'][sim, :].sum() +
                        maint_results['total_cost'][sim] +
                        env_results['total_env_cost'][sim]
                )

                # Total revenue
                total_revenue = econ_results['revenue'][sim, :].sum()

                # Calculate integrated NPV
                cash_flows = (
                        econ_results['revenue'][sim, :] -
                        econ_results['opex'][sim, :] -
                        maint_results['maintenance_cost'][sim, :] -
                        (env_results['emissions'][sim, :] +
                         env_results['water_treatment'][sim, :])
                )
                discount_rate = 0.1
                discount_factors = 1 / (1 + discount_rate) ** np.arange(forecast_days)
                integrated_npv = np.sum(cash_flows * discount_factors)

                # Store integrated metrics
                if 'total_cost' not in integrated_results['integrated_metrics']:
                    integrated_results['integrated_metrics']['total_cost'] = []
                    integrated_results['integrated_metrics']['total_revenue'] = []
                    integrated_results['integrated_metrics']['integrated_npv'] = []

                integrated_results['integrated_metrics']['total_cost'].append(total_cost)
                integrated_results['integrated_metrics']['total_revenue'].append(total_revenue)
                integrated_results['integrated_metrics']['integrated_npv'].append(integrated_npv)

            # Convert lists to numpy arrays
            for key in integrated_results['integrated_metrics']:
                integrated_results['integrated_metrics'][key] = np.array(
                    integrated_results['integrated_metrics'][key]
                )

            return integrated_results

        except Exception as e:
            logger.error(f"Error in integrated simulation: {str(e)}")
            return {}

    def calculate_statistics(self, results: Dict) -> Dict:
        """Calculate statistics for simulation results"""
        try:
            stats = {}

            def calc_stats(data: np.ndarray) -> Dict:
                return {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'p10': np.percentile(data, 10),
                    'p50': np.percentile(data, 50),
                    'p90': np.percentile(data, 90)
                }

            # Calculate statistics for each result type
            for category, data in results.items():
                if isinstance(data, dict):
                    stats[category] = {}
                    for subcategory, values in data.items():
                        if isinstance(values, np.ndarray):
                            if values.ndim == 1:
                                stats[category][subcategory] = calc_stats(values)
                            elif values.ndim == 2:
                                stats[category][subcategory] = {
                                    'daily': calc_stats(values.mean(axis=0)),
                                    'total': calc_stats(values.sum(axis=1))
                                }
                elif isinstance(data, np.ndarray):
                    stats[category] = calc_stats(data)

            return stats

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}

    def generate_monte_carlo_report(self, well_data: WellData,
                                    n_simulations: int = 1000,
                                    forecast_days: int = 365) -> Dict:
        """Generate comprehensive Monte Carlo simulation report"""
        try:
            # Run integrated simulation
            results = self.run_integrated_simulation(
                well_data, n_simulations, forecast_days)

            # Calculate statistics
            stats = self.calculate_statistics(results)

            # Generate risk metrics
            risk_metrics = {
                'npv_at_risk': abs(stats['economics']['npv']['p10']),
                'production_uncertainty': (
                                                  stats['production']['oil_production']['total']['p90'] -
                                                  stats['production']['oil_production']['total']['p10']
                                          ) / stats['production']['oil_production']['total']['p50'],
                'cost_uncertainty': (
                                            stats['integrated_metrics']['total_cost']['p90'] -
                                            stats['integrated_metrics']['total_cost']['p10']
                                    ) / stats['integrated_metrics']['total_cost']['p50']
            }

            # Generate recommendations
            recommendations = []
            if risk_metrics['npv_at_risk'] > 1e6:  # Example threshold
                recommendations.append(
                    "High NPV risk - consider risk mitigation strategies"
                )
            if risk_metrics['production_uncertainty'] > 0.5:  # Example threshold
                recommendations.append(
                    "High production uncertainty - review decline assumptions"
                )
            if risk_metrics['cost_uncertainty'] > 0.4:  # Example threshold
                recommendations.append(
                    "High cost uncertainty - review cost estimates"
                )

            report = {
                'simulation_results': results,
                'statistics': stats,
                'risk_metrics': risk_metrics,
                'recommendations': recommendations,
                'simulation_params': {
                    'n_simulations': n_simulations,
                    'forecast_days': forecast_days,
                    'run_date': datetime.now().isoformat()
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating Monte Carlo report: {str(e)}")
            return {}


