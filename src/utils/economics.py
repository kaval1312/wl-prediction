import numpy as np
from typing import Dict, List, Optional

class EconomicAnalyzer:
    def __init__(self, tax_rate: float = 0.21, discount_rate: float = 0.1):
        self.tax_rate = tax_rate
        self.discount_rate = discount_rate

    def calculate_npv(
        self,
        cash_flows: np.ndarray,
        initial_investment: float
    ) -> float:
        """
        Calculate Net Present Value
        """
        periods = np.arange(len(cash_flows))
        discounted_flows = cash_flows / (1 + self.discount_rate) ** periods
        return -initial_investment + np.sum(discounted_flows)

    def calculate_irr(
        self,
        cash_flows: np.ndarray,
        initial_investment: float
    ) -> Optional[float]:
        """
        Calculate Internal Rate of Return
        """
        try:
            flow_series = np.array([-initial_investment] + list(cash_flows))
            return np.irr(flow_series)
        except:
            return None

    def calculate_scenarios(
        self,
        production: np.ndarray,
        oil_price: float,
        opex: float,
        initial_investment: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate different economic scenarios
        """
        scenarios = {
            'base': {'price_factor': 1.0, 'cost_factor': 1.0},
            'optimistic': {'price_factor': 1.2, 'cost_factor': 0.9},
            'pessimistic': {'price_factor': 0.8, 'cost_factor': 1.1}
        }
        
        results = {}
        for scenario, factors in scenarios.items():
            # Adjust prices and costs
            adj_price = oil_price * factors['price_factor']
            adj_opex = opex * factors['cost_factor']
            
            # Calculate cash flows
            revenue = production * adj_price
            costs = production * adj_opex
            taxable_income = revenue - costs
            taxes = taxable_income * self.tax_rate
            cash_flow = revenue - costs - taxes
            
            # Calculate metrics
            npv = self.calculate_npv(cash_flow, initial_investment)
            irr = self.calculate_irr(cash_flow, initial_investment)
            
            results[scenario] = {
                'npv': npv,
                'irr': irr,
                'total_revenue': np.sum(revenue),
                'total_costs': np.sum(costs),
                'total_taxes': np.sum(taxes),
                'net_cash_flow': np.sum(cash_flow)
            }
        
        return results