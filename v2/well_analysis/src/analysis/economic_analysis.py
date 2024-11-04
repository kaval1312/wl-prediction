import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
from ..core.data_structures import WellData
from ..core.constants import FINANCIAL_CATEGORIES

logger = logging.getLogger(__name__)


class EconomicAnalyzer:
    def __init__(self):
        """Initialize economic analyzer"""
        self.results = {}

    def calculate_npv(self, cash_flows: np.ndarray,
                      discount_rate: float) -> float:
        """Calculate Net Present Value"""
        try:
            periods = np.arange(len(cash_flows))
            return np.sum(cash_flows / (1 + discount_rate) ** periods)
        except Exception as e:
            logger.error(f"Error calculating NPV: {str(e)}")
            return 0.0

    def calculate_irr(self, cash_flows: np.ndarray) -> Optional[float]:
        """Calculate Internal Rate of Return"""
        try:
            # Add initial investment as negative cash flow
            if cash_flows[0] >= 0:
                cash_flows = np.insert(cash_flows, 0, -cash_flows[0])

            # Find IRR using numpy's financial functions
            return np.irr(cash_flows)
        except Exception as e:
            logger.error(f"Error calculating IRR: {str(e)}")
            return None

    def analyze_economics(self, well_data: WellData,
                          oil_price: float,
                          gas_price: float,
                          discount_rate: float = 0.1) -> Dict:
        """Perform economic analysis for a well"""
        try:
            # Get production and financial data
            prod_data = well_data.production_data
            fin_data = well_data.financial_data

            # Calculate revenues
            revenues = (
                    prod_data['Oil_Production_BBL'] * oil_price +
                    prod_data['Gas_Production_MCF'] * gas_price
            )

            # Calculate costs
            opex = fin_data[[col for col in fin_data.columns
                             if 'Operations' in col]].sum(axis=1)
            maintenance = fin_data[[col for col in fin_data.columns
                                    if 'Maintenance' in col]].sum(axis=1)

            # Calculate cash flows
            cash_flows = revenues - opex - maintenance

            # Calculate economic metrics
            npv = self.calculate_npv(cash_flows.values, discount_rate)
            irr = self.calculate_irr(cash_flows.values)
            # Calculate additional metrics
            cumulative_revenue = revenues.sum()
            cumulative_opex = opex.sum()
            cumulative_maintenance = maintenance.sum()
            net_income = cumulative_revenue - cumulative_opex - cumulative_maintenance

            # Calculate unit costs
            total_oil = prod_data['Oil_Production_BBL'].sum()
            total_gas = prod_data['Gas_Production_MCF'].sum()

            opex_per_boe = (cumulative_opex + cumulative_maintenance) / (
                    total_oil + total_gas / 6)  # Convert MCF to BOE

            # Calculate payout period
            cumulative_cash_flow = cash_flows.cumsum()
            payout_period = np.where(cumulative_cash_flow > 0)[0]
            payout_days = len(prod_data) if len(payout_period) == 0 else payout_period[0]

            # Calculate profitability metrics
            revenue_per_boe = cumulative_revenue / (total_oil + total_gas / 6)
            profit_margin = net_income / cumulative_revenue if cumulative_revenue > 0 else 0

            return {
                'npv': npv,
                'irr': irr,
                'cumulative_revenue': cumulative_revenue,
                'cumulative_opex': cumulative_opex,
                'cumulative_maintenance': cumulative_maintenance,
                'net_income': net_income,
                'opex_per_boe': opex_per_boe,
                'revenue_per_boe': revenue_per_boe,
                'profit_margin': profit_margin,
                'payout_days': payout_days,
                'daily_cash_flows': cash_flows.tolist(),
                'cumulative_cash_flows': cumulative_cash_flow.tolist()
            }

        except Exception as e:
            logger.error(f"Error in economic analysis: {str(e)}")
            return {}

    def calculate_sensitivity(self, well_data: WellData,
                              base_oil_price: float,
                              base_gas_price: float,
                              variations: List[float] = [-0.2, -0.1, 0, 0.1, 0.2]
                              ) -> Dict[str, List[Dict]]:
        """Perform sensitivity analysis on economic parameters"""
        try:
            sensitivity_results = {
                'oil_price': [],
                'gas_price': [],
                'opex': [],
                'discount_rate': []
            }

            # Base case analysis
            base_case = self.analyze_economics(
                well_data, base_oil_price, base_gas_price)

            # Sensitivity to oil price
            for var in variations:
                test_price = base_oil_price * (1 + var)
                result = self.analyze_economics(
                    well_data, test_price, base_gas_price)
                sensitivity_results['oil_price'].append({
                    'variation': var,
                    'npv_change': (result['npv'] - base_case['npv']) / base_case['npv']
                })

            # Sensitivity to gas price
            for var in variations:
                test_price = base_gas_price * (1 + var)
                result = self.analyze_economics(
                    well_data, base_oil_price, test_price)
                sensitivity_results['gas_price'].append({
                    'variation': var,
                    'npv_change': (result['npv'] - base_case['npv']) / base_case['npv']
                })

            return sensitivity_results

        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            return {}

    def forecast_economics(self, well_data: WellData,
                           forecast_days: int,
                           oil_price: float,
                           gas_price: float,
                           discount_rate: float = 0.1) -> Dict:
        """Forecast future economics"""
        try:
            from ..analysis.decline_analysis import DeclineAnalyzer

            # Get production forecast
            decline_analyzer = DeclineAnalyzer()
            oil_forecast = decline_analyzer.forecast_production(
                well_data, forecast_days, 'oil')
            gas_forecast = decline_analyzer.forecast_production(
                well_data, forecast_days, 'gas')

            # Calculate forecasted revenues
            forecast_revenues = (
                    oil_forecast['Oil_Rate'] * oil_price +
                    gas_forecast['Gas_Rate'] * gas_price
            )

            # Estimate forecasted costs
            current_opex = well_data.financial_data[[col for col in
                                                     well_data.financial_data.columns if 'Operations' in col]].sum(
                axis=1).mean()
            forecast_opex = np.full(forecast_days, current_opex)

            # Calculate forecasted cash flows
            forecast_cash_flows = forecast_revenues - forecast_opex

            # Calculate economic metrics for forecast
            forecast_npv = self.calculate_npv(forecast_cash_flows.values, discount_rate)
            forecast_irr = self.calculate_irr(forecast_cash_flows.values)

            return {
                'forecast_npv': forecast_npv,
                'forecast_irr': forecast_irr,
                'forecast_revenues': forecast_revenues.sum(),
                'forecast_opex': forecast_opex.sum(),
                'forecast_net_income': forecast_revenues.sum() - forecast_opex.sum(),
                'daily_forecast_cash_flows': forecast_cash_flows.tolist()
            }

        except Exception as e:
            logger.error(f"Error in economic forecasting: {str(e)}")
            return {}

    def calculate_breakeven_prices(self, well_data: WellData,
                                   min_price: float = 0,
                                   max_price: float = 200,
                                   steps: int = 100) -> Dict[str, float]:
        """Calculate breakeven prices for oil and gas"""
        try:
            # Function to find NPV at given prices
            def calc_npv(oil_price, gas_price):
                result = self.analyze_economics(well_data, oil_price, gas_price)
                return result.get('npv', 0)

            # Find oil breakeven (holding gas price constant)
            gas_price_avg = (min_price + max_price) / 2
            oil_prices = np.linspace(min_price, max_price, steps)
            npvs = [calc_npv(op, gas_price_avg) for op in oil_prices]
            oil_breakeven = np.interp(0, npvs, oil_prices)

            # Find gas breakeven (holding oil price constant)
            oil_price_avg = (min_price + max_price) / 2
            gas_prices = np.linspace(min_price, max_price, steps)
            npvs = [calc_npv(oil_price_avg, gp) for gp in gas_prices]
            gas_breakeven = np.interp(0, npvs, gas_prices)

            return {
                'oil_breakeven': oil_breakeven,
                'gas_breakeven': gas_breakeven
            }

        except Exception as e:
            logger.error(f"Error calculating breakeven prices: {str(e)}")
            return {}

    def analyze_cost_structure(self, well_data: WellData) -> Dict:
        """Analyze the cost structure of the well"""
        try:
            fin_data = well_data.financial_data

            # Categorize costs
            cost_categories = {}

            for category in FINANCIAL_CATEGORIES:
                category_cols = [col for col in fin_data.columns
                                 if col.startswith(category)]
                cost_categories[category] = fin_data[category_cols].sum().sum()

            # Calculate percentages
            total_cost = sum(cost_categories.values())
            cost_percentages = {
                cat: (cost / total_cost) if total_cost > 0 else 0
                for cat, cost in cost_categories.items()
            }

            # Calculate metrics per BOE
            total_oil = well_data.production_data['Oil_Production_BBL'].sum()
            total_gas = well_data.production_data['Gas_Production_MCF'].sum()
            total_boe = total_oil + total_gas / 6

            cost_per_boe = {
                cat: (cost / total_boe) if total_boe > 0 else 0
                for cat, cost in cost_categories.items()
            }

            return {
                'cost_categories': cost_categories,
                'cost_percentages': cost_percentages,
                'cost_per_boe': cost_per_boe,
                'total_cost': total_cost,
                'total_boe': total_boe
            }

        except Exception as e:
            logger.error(f"Error analyzing cost structure: {str(e)}")
            return {}

    def calculate_unit_metrics(self, well_data: WellData) -> Dict:
        """Calculate various unit-based economic metrics"""
        try:
            # Get production data
            total_oil = well_data.production_data['Oil_Production_BBL'].sum()
            total_gas = well_data.production_data['Gas_Production_MCF'].sum()
            total_boe = total_oil + total_gas / 6

            # Get financial data
            fin_data = well_data.financial_data

            # Calculate various unit metrics
            opex = fin_data[[col for col in fin_data.columns
                             if 'Operations' in col]].sum().sum()
            maintenance = fin_data[[col for col in fin_data.columns
                                    if 'Maintenance' in col]].sum().sum()
            labor = fin_data[[col for col in fin_data.columns
                              if 'Labor' in col]].sum().sum()

            metrics = {
                'opex_per_boe': opex / total_boe if total_boe > 0 else 0,
                'maintenance_per_boe': maintenance / total_boe if total_boe > 0 else 0,
                'labor_per_boe': labor / total_boe if total_boe > 0 else 0,
                'total_cost_per_boe': (opex + maintenance) / total_boe if total_boe > 0 else 0,
                'opex_per_day': opex / len(well_data.production_data),
                'maintenance_per_day': maintenance / len(well_data.production_data)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating unit metrics: {str(e)}")
            return {}