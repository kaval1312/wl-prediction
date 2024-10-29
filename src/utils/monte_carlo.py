import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from .lease_terms import LeaseTerms, AbandonmentCosts


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
            initial_rate: float,
            decline_rate: float,
            oil_price: float,
            opex: float,
            initial_investment: float,
            lease_terms: LeaseTerms,
            abandonment_costs: AbandonmentCosts,
            months: int = 120,
            iterations: int = 1000,
            confidence_level: float = 0.90
    ) -> MonteCarloResults:
        """Run comprehensive Monte Carlo analysis"""
        # Initialize arrays
        production_profiles = np.zeros((iterations, months))
        npv_values = np.zeros(iterations)
        roi_values = np.zeros(iterations)
        payback_values = np.zeros(iterations)

        # Run simulations
        for i in range(iterations):
            # Generate random variations
            prod_uncertainty = self.rng.normal(1, 0.2)
            price_uncertainty = self.rng.normal(1, 0.15)
            cost_uncertainty = self.rng.normal(1, 0.1)

            # Calculate production profile
            time = np.arange(months)
            production = initial_rate * prod_uncertainty * np.exp(-decline_rate * time)
            production_profiles[i] = production

            # Calculate economics
            revenue = production * oil_price * price_uncertainty
            costs = production * opex * cost_uncertainty
            cash_flow = revenue - costs

            # Calculate metrics
            npv = -initial_investment + np.sum(cash_flow / (1 + 0.1) ** (time / 12))
            npv_values[i] = npv

            # Calculate ROI
            total_revenue = np.sum(revenue)
            total_costs = np.sum(costs) + initial_investment
            roi_values[i] = ((total_revenue - total_costs) / initial_investment) * 100

            # Calculate payback
            cumulative_cash_flow = np.cumsum(cash_flow)
            payback_periods = np.where(cumulative_cash_flow >= initial_investment)[0]
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

        # Calculate risk metrics
        risk_metrics = {
            'probability_of_loss': np.mean(npv_values < 0),
            'value_at_risk': np.percentile(npv_values, 5),
            'expected_shortfall': np.mean(npv_values[npv_values < np.percentile(npv_values, 5)]),
            'probability_of_target_roi': np.mean(roi_values > 15),  # 15% target ROI
            'average_payback': np.mean(payback_values)
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
            statistics={
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
            },
            risk_metrics=risk_metrics,
            detailed_results=detailed_results
        )