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
    
    monthly_rate = (1 + discount_rate) ** (1/12) - 1
    present_values = cash_flows / (1 + monthly_rate) ** time_periods
    
    return -initial_investment + np.sum(present_values)

def calculate_irr(
    cash_flows: np.ndarray,
    initial_investment: float,
    tolerance: float = 1e-6
) -> Optional[float]:
    """Calculate Internal Rate of Return."""
    def try_rate(rate):
        return calculate_npv(cash_flows, rate, initial_investment)
    
    # Binary search for IRR
    low_rate = -0.99
    high_rate = 10.0
    
    for _ in range(1000):  # Maximum iterations
        rate = (low_rate + high_rate) / 2
        npv = try_rate(rate)
        
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
        monthly_rate = (1 + discount_rate) ** (1/12) - 1
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

def calculate_unit_costs(
    total_costs: np.ndarray,
    production: np.ndarray,
    cost_categories: Optional[Dict[str, np.ndarray]] = None
) -> Union[float, Dict[str, float]]:
    """Calculate unit costs."""
    total_production = np.sum(production)
    
    if total_production == 0:
        return float('inf')
    
    if cost_categories is None:
        return np.sum(total_costs) / total_production
    
    unit_costs = {}
    for category, costs in cost_categories.items():
        unit_costs[category] = np.sum(costs) / total_production
    
    unit_costs['total'] = np.sum(total_costs) / total_production
    return unit_costs

def calculate_profitability_index(
    npv: float,
    initial_investment: float
) -> float:
    """Calculate Profitability Index."""
    if initial_investment == 0:
        return float('inf') if npv > 0 else float('-inf')
    return 1 + (npv / initial_investment)

def calculate_break_even_price(
    production: np.ndarray,
    fixed_costs: np.ndarray,
    variable_costs_per_bbl: float,
    discount_rate: float = 0.1
) -> float:
    """Calculate break-even oil price."""
    monthly_rate = (1 + discount_rate) ** (1/12) - 1
    time = np.arange(len(production))
    
    pv_production = np.sum(production / (1 + monthly_rate) ** time)
    pv_fixed_costs = np.sum(fixed_costs / (1 + monthly_rate) ** time)
    
    if pv_production == 0:
        return float('inf')
    
    return (pv_fixed_costs / pv_production) + variable_costs_per_bbl

def sensitivity_analysis(
    base_case: Dict[str, float],
    variables: Dict[str, Tuple[float, float]],
    metric_function: callable
) -> Dict[str, List[Tuple[float, float]]]:
    """Perform sensitivity analysis."""
    results = {}
    
    for var_name, (min_val, max_val) in variables.items():
        variable_results = []
        test_values = np.linspace(min_val, max_val, 10)
        
        for test_value in test_values:
            test_case = base_case.copy()
            test_case[var_name] = test_value
            metric_value = metric_function(**test_case)
            variable_results.append((test_value, metric_value))
        
        results[var_name] = variable_results
    
    return results