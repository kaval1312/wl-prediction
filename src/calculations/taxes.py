from typing import Dict, List
import numpy as np

def calculate_tax_obligations(
    revenue: float,
    expenses: float,
    tax_rates: Dict[str, float],
    deductions: List[Dict],
    credits: List[Dict]
) -> float:
    """
    Calculate total tax obligations.
    
    Args:
        revenue: Gross revenue
        expenses: Total expenses
        tax_rates: Dictionary of tax rates
        deductions: List of applicable deductions
        credits: List of applicable credits
    
    Returns:
        Total tax obligation
    """
    taxable_income = revenue - expenses
    
    # Apply deductions
    for deduction in deductions:
        if deduction['type'] == 'percentage':
            taxable_income -= revenue * deduction['value']
        else:  # fixed amount
            taxable_income -= deduction['value']
    
    # Calculate base tax
    tax = taxable_income * tax_rates['federal']
    tax += taxable_income * tax_rates['state']
    
    # Apply credits
    for credit in credits:
        tax -= credit['value']
    
    return max(0, tax)

def calculate_depletion_allowance(
    revenue: float,
    rate: float = 0.15,
    limit: float = 0.65
) -> float:
    """
    Calculate percentage depletion allowance.
    
    Args:
        revenue: Gross revenue
        rate: Depletion rate (typically 15%)
        limit: Net income limitation (typically 65%)
    
    Returns:
        Depletion allowance amount
    """
    allowance = revenue * rate
    income_limit = revenue * limit
    return min(allowance, income_limit)

def calculate_severance_tax(
    production: float,
    price: float,
    tax_rate: float,
    exemptions: Dict = None
) -> float:
    """
    Calculate severance tax.
    
    Args:
        production: Oil production (barrels)
        price: Oil price per barrel
        tax_rate: Severance tax rate
        exemptions: Dictionary of applicable exemptions
    
    Returns:
        Severance tax amount
    """
    taxable_value = production * price
    
    if exemptions:
        for exemption in exemptions.values():
            taxable_value *= (1 - exemption)
    
    return taxable_value * tax_rate

def calculate_carbon_tax(
    production: float,
    emission_factor: float,
    tax_rate: float,
    carbon_credits: float = 0.0,
    offset_factor: float = 0.0
) -> float:
    """
    Calculate carbon tax based on production and emissions.
    
    Args:
        production: Oil production (barrels)
        emission_factor: CO2 emissions per barrel (tonnes CO2/bbl)
        tax_rate: Carbon tax rate ($/tonne CO2)
        carbon_credits: Available carbon credits (tonnes CO2)
        offset_factor: Emissions reduction factor from mitigation efforts
    
    Returns:
        Carbon tax amount
    """
    # Calculate total emissions
    total_emissions = production * emission_factor * (1 - offset_factor)
    
    # Subtract available carbon credits
    taxable_emissions = max(0, total_emissions - carbon_credits)
    
    # Calculate tax
    return taxable_emissions * tax_rate
