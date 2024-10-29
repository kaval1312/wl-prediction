from typing import Dict, List, Tuple
import numpy as np


def calculate_tax_obligations(
        revenue: np.ndarray,
        costs: np.ndarray,
        tax_rates: Dict[str, float],
        deductions: Dict[str, float],
        credits: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate total tax obligations.

    Args:
        revenue: Gross revenue array
        costs: Total costs array
        tax_rates: Dictionary of tax rates
        deductions: Dictionary of deductions
        credits: Dictionary of tax credits

    Returns:
        Tuple of (total_tax_obligations, tax_components)
    """
    # Calculate taxable income
    taxable_income = revenue - costs

    # Apply deductions
    for deduction_rate in deductions.values():
        taxable_income -= revenue * deduction_rate

    # Calculate tax components
    federal_tax = np.maximum(0, taxable_income * tax_rates.get('federal', 0.21))
    state_tax = np.maximum(0, taxable_income * tax_rates.get('state', 0.05))

    # Calculate severance tax
    severance_tax = calculate_severance_tax(
        revenue=revenue,
        rate=tax_rates.get('severance', 0.045)
    )

    # Apply credits
    total_credits = sum(credits.values())
    total_tax = federal_tax + state_tax + severance_tax
    adjusted_tax = np.maximum(0, total_tax - total_credits)

    tax_components = {
        'federal': federal_tax,
        'state': state_tax,
        'severance': severance_tax,
        'credits': np.full_like(federal_tax, total_credits),
        'final': adjusted_tax
    }

    return adjusted_tax, tax_components


def calculate_severance_tax(
        revenue: np.ndarray,
        rate: float,
        exemptions: Dict[str, float] = None
) -> np.ndarray:
    """Calculate severance tax."""
    taxable_value = revenue.copy()

    if exemptions:
        for exemption_rate in exemptions.values():
            taxable_value *= (1 - exemption_rate)

    return taxable_value * rate


def calculate_depletion_allowance(
        revenue: np.ndarray,
        rate: float = 0.15,
        net_income_limit: float = 0.65
) -> np.ndarray:
    """Calculate percentage depletion allowance."""
    allowance = revenue * rate
    income_limit = revenue * net_income_limit
    return np.minimum(allowance, income_limit)


def calculate_depreciation(
        initial_value: float,
        months: int,
        method: str = 'MACRS',
        asset_life: int = 7
) -> np.ndarray:
    """Calculate monthly depreciation."""
    if method == 'MACRS':
        # MACRS rates for different asset lives
        rates = {
            5: [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576],
            7: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893],
            10: [0.10, 0.18, 0.144, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655]
        }

        if asset_life not in rates:
            raise ValueError(f"Unsupported asset life: {asset_life}")

        annual_depreciation = np.zeros(asset_life)
        annual_depreciation[:len(rates[asset_life])] = rates[asset_life]

        # Convert to monthly
        monthly_depreciation = np.repeat(annual_depreciation, 12)[:months]
        return monthly_depreciation * initial_value / 12

    elif method == 'straight_line':
        monthly_rate = 1 / (asset_life * 12)
        return np.full(months, initial_value * monthly_rate)

    else:
        raise ValueError(f"Unsupported depreciation method: {method}")


def calculate_tax_basis(
        initial_investment: float,
        accumulated_depreciation: np.ndarray
) -> np.ndarray:
    """Calculate remaining tax basis."""
    return np.maximum(0, initial_investment - np.cumsum(accumulated_depreciation))


def analyze_tax_scenarios(
        revenue: np.ndarray,
        costs: np.ndarray,
        base_rates: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Analyze different tax scenarios."""
    results = {}

    # Base case
    base_tax, base_components = calculate_tax_obligations(
        revenue=revenue,
        costs=costs,
        tax_rates=base_rates,
        deductions={},
        credits={}
    )
    results['base'] = base_components

    # Alternative scenarios
    for name, rate_adjustments in scenarios.items():
        adjusted_rates = base_rates.copy()
        adjusted_rates.update(rate_adjustments)

        scenario_tax, scenario_components = calculate_tax_obligations(
            revenue=revenue,
            costs=costs,
            tax_rates=adjusted_rates,
            deductions={},
            credits={}
        )
        results[name] = scenario_components

    return results