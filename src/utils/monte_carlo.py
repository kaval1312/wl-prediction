# src/utils/monte_carlo.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .economic_parameters import EconomicParameters


@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results"""
    production_profiles: np.ndarray
    npv_distribution: np.ndarray
    roi_distribution: np.ndarray
    payback_distribution: np.ndarray
    percentiles: Dict[str, Dict[str, float]]
    statistics: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    detailed_results: pd.DataFrame


class MonteCarloSimulator:
    """Monte Carlo simulator for production forecasting"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize simulator with optional seed"""
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()

    def run_full_analysis(
            self,
            economic_params: EconomicParameters,
            months: int = 120,
            iterations: int = 1000,
            confidence_level: float = 0.90
    ) -> MonteCarloResults:
        """
        Run comprehensive Monte Carlo analysis.

        Args:
            economic_params: Economic parameters for analysis
            months: Number of months to simulate
            iterations: Number of Monte Carlo iterations
            confidence_level: Confidence level for risk metrics

        Returns:
            MonteCarloResults object containing all simulation results
        """
        # Initialize arrays for results
        production_profiles = np.zeros((iterations, months))
        npv_values = np.zeros(iterations)
        roi_values = np.zeros(iterations)
        payback_values = np.zeros(iterations)

        # Uncertainty parameters
        production_uncertainty = 0.20  # 20% standard deviation
        price_uncertainty = 0.15  # 15% standard deviation
        cost_uncertainty = 0.10  # 10% standard deviation

        # Run simulations
        for i in range(iterations):
            # Generate random variations
            prod_factor = self.rng.normal(1, production_uncertainty)
            price_factor = self.rng.normal(1, price_uncertainty)
            cost_factor = self.rng.normal(1, cost_uncertainty)

            # Calculate production profile
            time = np.arange(months)
            production = economic_params.initial_rate * prod_factor * \
                         np.exp(-economic_params.decline_rate * time)
            production_profiles[i] = production

            # Calculate economics
            revenue = production * economic_params.oil_price * price_factor
            costs = production * economic_params.opex * cost_factor

            # Apply working and net revenue interests
            net_revenue = revenue * economic_params.net_revenue_interest
            working_interest_costs = costs * economic_params.working_interest

            # Calculate monthly cash flow
            cash_flow = net_revenue - working_interest_costs

            # Calculate NPV
            monthly_discount_rate = (1 + economic_params.discount_rate) ** (1 / 12) - 1
            discount_factors = 1 / (1 + monthly_discount_rate) ** time
            npv = -economic_params.initial_investment + np.sum(cash_flow * discount_factors)
            npv_values[i] = npv

            # Calculate ROI
            total_revenue = np.sum(net_revenue)
            total_costs = np.sum(working_interest_costs) + economic_params.initial_investment
            roi = ((total_revenue - total_costs) / economic_params.initial_investment) * 100
            roi_values[i] = roi

            # Calculate payback period
            cumulative_cash_flow = np.cumsum(cash_flow)
            payback_periods = np.where(cumulative_cash_flow >= economic_params.initial_investment)[0]
            payback_values[i] = len(time) if len(payback_periods) == 0 else payback_periods[0]

        # Calculate percentiles
        percentiles = {
            'Production': {
                'P10': np.percentile(production_profiles[:, -1], 10),
                'P50': np.percentile(production_profiles[:, -1], 50),
                'P90': np.percentile(production_profiles[:, -1], 90)
            },
            'NPV': {
                'P10': np.percentile(npv_values, 10),
                'P50': np.percentile(npv_values, 50),
                'P90': np.percentile(npv_values, 90)
            },
            'ROI': {
                'P10': np.percentile(roi_values, 10),
                'P50': np.percentile(roi_values, 50),
                'P90': np.percentile(roi_values, 90)
            }
        }

        # Calculate statistics
        statistics = {
            'Production': {
                'mean': np.mean(production_profiles[:, -1]),
                'std': np.std(production_profiles[:, -1]),
                'min': np.min(production_profiles[:, -1]),
                'max': np.max(production_profiles[:, -1])
            },
            'NPV': {
                'mean': np.mean(npv_values),
                'std': np.std(npv_values),
                'min': np.min(npv_values),
                'max': np.max(npv_values)
            },
            'ROI': {
                'mean': np.mean(roi_values),
                'std': np.std(roi_values),
                'min': np.min(roi_values),
                'max': np.max(roi_values)
            }
        }

        # Calculate risk metrics
        var_level = 1 - confidence_level
        risk_metrics = {
            'probability_of_loss': np.mean(npv_values < 0),
            'value_at_risk': np.percentile(npv_values, var_level * 100),
            'expected_shortfall': np.mean(npv_values[npv_values < np.percentile(npv_values, var_level * 100)]),
            'probability_of_target_roi': np.mean(roi_values > 15)  # 15% target ROI
        }

        # Create detailed results DataFrame
        detailed_results = pd.DataFrame({
            'Iteration': range(iterations),
            'Final_Production': production_profiles[:, -1],
            'NPV': npv_values,
            'ROI': roi_values,
            'Payback_Months': payback_values,
            'Profitable': npv_values > 0,
            'Meets_ROI_Target': roi_values > 15
        })

        return MonteCarloResults(
            production_profiles=production_profiles,
            npv_distribution=npv_values,
            roi_distribution=roi_values,
            payback_distribution=payback_values,
            percentiles=percentiles,
            statistics=statistics,
            risk_metrics=risk_metrics,
            detailed_results=detailed_results
        )

    def analyze_sensitivity(
            self,
            base_params: EconomicParameters,
            variables: Dict[str, Tuple[float, float]],
            months: int = 120,
            points: int = 10
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Perform sensitivity analysis on economic parameters.

        Args:
            base_params: Base case economic parameters
            variables: Dictionary of variables and their ranges to test
            months: Number of months to simulate
            points: Number of points to test for each variable

        Returns:
            Dictionary of sensitivity results
        """
        results = {}
        base_results = self.run_full_analysis(base_params, months)
        base_npv = base_results.statistics['NPV']['mean']

        for var_name, (min_val, max_val) in variables.items():
            variable_results = []
            test_values = np.linspace(min_val, max_val, points)

            for test_value in test_values:
                # Create modified parameters
                test_params = base_params.copy()
                setattr(test_params, var_name.lower(), test_value)

                # Run analysis with test parameters
                test_results = self.run_full_analysis(test_params, months)
                test_npv = test_results.statistics['NPV']['mean']

                # Calculate change from base case
                npv_change = (test_npv - base_npv) / base_npv * 100
                variable_results.append((test_value, npv_change))

            results[var_name] = variable_results

        return results