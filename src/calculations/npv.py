import numpy as np
from typing import List, Tuple, Optional, Dict, Union


def calculate_npv(
        cash_flows: np.ndarray,
        discount_rate: float = 0.1,
        initial_investment: float = 0.0,
        time_periods: Optional[np.ndarray] = None
) -> float:
    """Calculate Net Present Value of cash flows."""
    if time_periods is None:
        time_periods = np.arange(len(cash_flows))

    monthly_rate = (1 + discount_rate) ** (1 / 12) - 1
    present_values = cash_flows / (1 + monthly_rate) ** time_periods

    return -initial_investment + np.sum(present_values)


def calculate_irr(
        cash_flows: np.ndarray,
        initial_investment: float,
        tolerance: float = 1e-6,
        max_iterations: int = 1000
) -> Optional[float]:
    """Calculate Internal Rate of Return."""

    def npv_at_rate(rate):
        return calculate_npv(cash_flows, rate, initial_investment)

    # Binary search for IRR
    low_rate = -0.99
    high_rate = 10.0

    for _ in range(max_iterations):
        rate = (low_rate + high_rate) / 2
        npv = npv_at_rate(rate)

        if abs(npv) < tolerance:
            return rate
        elif npv > 0:
            low_rate = rate
        else:
            high_rate = rate

        if high_rate - low_rate < tolerance:
            return rate

    return None


def calculate_payback_period(
        cash_flows: np.ndarray,
        initial_investment: float,
        discount_rate: Optional[float] = None
) -> Tuple[float, bool]:
    """Calculate payback period."""
    if discount_rate:
        monthly_rate = (1 + discount_rate) ** (1 / 12) - 1
        flows = cash_flows / (1 + monthly_rate) ** np.arange(len(cash_flows))
    else:
        flows = cash_flows

    cumulative = np.cumsum(flows)
    payback_periods = np.where(cumulative >= initial_investment)[0]

    if len(payback_periods) == 0:
        return float('inf'), False

    first_period = payback_periods[0]
    if first_period == 0:
        return 0.0, True

    # Interpolate
    if first_period > 0:
        prev_cumulative = cumulative[first_period - 1]
        period_flow = cumulative[first_period] - prev_cumulative
        fraction = (initial_investment - prev_cumulative) / period_flow
        return first_period - 1 + fraction, True

    return first_period, True


def calculate_break_even_price(
        production: np.ndarray,
        fixed_costs: np.ndarray,
        variable_costs_per_bbl: float,
        discount_rate: float = 0.1,
        target_npv: float = 0.0,
        tolerance: float = 0.01
) -> float:
    """Calculate break-even oil price."""
    monthly_rate = (1 + discount_rate) ** (1 / 12) - 1
    time = np.arange(len(production))

    pv_production = np.sum(production / (1 + monthly_rate) ** time)
    pv_fixed_costs = np.sum(fixed_costs / (1 + monthly_rate) ** time)

    if pv_production == 0:
        return float('inf')

    # Price needed for NPV = target_npv
    break_even = (target_npv + pv_fixed_costs) / pv_production + variable_costs_per_bbl

    return max(0, break_even)


def calculate_profitability_metrics(
        production: np.ndarray,
        costs: np.ndarray,
        oil_price: float,
        initial_investment: float,
        discount_rate: float = 0.1
) -> Dict[str, float]:
    """Calculate comprehensive profitability metrics."""
    revenue = production * oil_price
    cash_flows = revenue - costs

    npv = calculate_npv(cash_flows, discount_rate, initial_investment)
    irr = calculate_irr(cash_flows, initial_investment)
    payback, has_payback = calculate_payback_period(cash_flows, initial_investment)

    # Calculate profit metrics
    total_revenue = np.sum(revenue)
    total_costs = np.sum(costs)
    net_profit = total_revenue - total_costs - initial_investment
    profit_margin = net_profit / total_revenue if total_revenue > 0 else 0
    roi = net_profit / initial_investment if initial_investment > 0 else float('inf')

    return {
        'npv': npv,
        'irr': irr if irr is not None else 0,
        'payback_period': payback if has_payback else float('inf'),
        'total_revenue': total_revenue,
        'total_costs': total_costs,
        'net_profit': net_profit,
        'profit_margin': profit_margin,
        'roi': roi,
        'unit_cost': total_costs / np.sum(production) if np.sum(production) > 0 else float('inf')
    }


def sensitivity_analysis(
        base_case: Dict[str, float],
        variables: Dict[str, Tuple[float, float]],
        metric_function: callable,
        points: int = 10
) -> Dict[str, List[Tuple[float, float]]]:
    """Perform sensitivity analysis on economic parameters."""
    results = {}

    for var_name, (min_val, max_val) in variables.items():
        variable_results = []
        test_values = np.linspace(min_val, max_val, points)

        for test_value in test_values:
            test_case = base_case.copy()
            test_case[var_name] = test_value
            metric_value = metric_function(**test_case)
            variable_results.append((test_value, metric_value))

        results[var_name] = variable_results

    return results