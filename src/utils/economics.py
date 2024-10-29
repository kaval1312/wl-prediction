# src/utils/economics.py
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
logger.debug("Importing economics module")
from typing import Dict, List, Optional, Tuple


from .economic_parameters import EconomicParameters

logger.debug(f"Imported EconomicParameters v{EconomicParameters.VERSION} from economic_parameters.py")


@dataclass
class EconomicResults:
    """Container for economic calculation results"""
    npv: float
    roi: float
    payback_period: float
    total_revenue: float
    total_costs: float
    net_profit: float
    profit_margins: np.ndarray
    cumulative_cash_flow: np.ndarray
    monthly_cash_flow: np.ndarray


class EconomicAnalyzer:
    """Enhanced economic analysis with sensitivity calculations"""

    def __init__(self, tax_rate: float = 0.21):
        self.tax_rate = tax_rate

    def calculate_metrics(
            self,
            production: np.ndarray,
            params: EconomicParameters
    ) -> EconomicResults:
        """Calculate all economic metrics"""
        time = np.arange(len(production))
        monthly_rate = (1 + params.discount_rate) ** (1 / 12) - 1

        # Calculate revenue and costs
        revenue = production * params.oil_price
        operating_costs = production * params.opex
        cash_flow = revenue - operating_costs
        cumulative_cash_flow = np.cumsum(cash_flow)

        # Calculate NPV
        npv = -params.initial_investment + np.sum(cash_flow / (1 + monthly_rate) ** time)

        # Calculate ROI metrics
        total_revenue = np.sum(revenue)
        total_costs = np.sum(operating_costs) + params.initial_investment
        net_profit = total_revenue - total_costs
        roi = (net_profit / params.initial_investment) * 100 if params.initial_investment > 0 else 0

        # Calculate payback period
        payback_periods = np.where(cumulative_cash_flow >= params.initial_investment)[0]
        payback_period = len(production) if len(payback_periods) == 0 else payback_periods[0]

        # Calculate profit margins
        profit_margins = (revenue - operating_costs) / revenue * 100

        return EconomicResults(
            npv=npv,
            roi=roi,
            payback_period=payback_period,
            total_revenue=total_revenue,
            total_costs=total_costs,
            net_profit=net_profit,
            profit_margins=profit_margins,
            cumulative_cash_flow=cumulative_cash_flow,
            monthly_cash_flow=cash_flow
        )

    def run_sensitivity_analysis(
            self,
            production: np.ndarray,
            params: EconomicParameters,
            variables: Dict[str, Tuple[float, float]],
            points: int = 10
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Perform sensitivity analysis on economic parameters.

        Args:
            production: Production profile
            params: Base economic parameters
            variables: Dictionary of variables and their ranges
            points: Number of points to calculate

        Returns:
            Dictionary of sensitivity results
        """
        results = {}
        base_results = self.calculate_metrics(production, params)

        for var_name, (min_val, max_val) in variables.items():
            variable_results = []
            test_values = np.linspace(min_val, max_val, points)

            for test_value in test_values:
                # Create modified parameters
                test_params = EconomicParameters(
                    oil_price=test_value if var_name == 'Oil Price' else params.oil_price,
                    opex=test_value if var_name == 'Operating Cost' else params.opex,
                    initial_investment=params.initial_investment,
                    discount_rate=params.discount_rate,
                    initial_rate=params.initial_rate,
                    decline_rate=test_value if var_name == 'Decline Rate' else params.decline_rate
                )

                # Calculate metrics with test parameters
                if var_name == 'Initial Production':
                    test_production = production * (test_value / production[0])
                elif var_name == 'Decline Rate':
                    time = np.arange(len(production))
                    test_production = params.initial_rate * np.exp(-test_value * time)
                else:
                    test_production = production

                test_results = self.calculate_metrics(test_production, test_params)
                variable_results.append((test_value, test_results.npv))

            results[var_name] = variable_results

        return results

    def create_scenarios(
            self,
            base_price: float,
            base_opex: float
    ) -> Dict[str, EconomicParameters]:
        """Create standard economic scenarios"""
        return {
            'base': EconomicParameters(
                oil_price=base_price,
                opex=base_opex,
                initial_investment=0,
                discount_rate=0.1,
                initial_rate=0,
                decline_rate=0
            ),
            'optimistic': EconomicParameters(
                oil_price=base_price * 1.2,
                opex=base_opex * 0.9,
                initial_investment=0,
                discount_rate=0.1,
                initial_rate=0,
                decline_rate=0
            ),
            'pessimistic': EconomicParameters(
                oil_price=base_price * 0.8,
                opex=base_opex * 1.1,
                initial_investment=0,
                discount_rate=0.12,
                initial_rate=0,
                decline_rate=0
            )
        }