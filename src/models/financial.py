from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

@dataclass
class FinancialMetrics:
    """Container for financial metrics."""
    npv: float
    irr: float
    payback_period: float
    roi: float
    break_even_price: float
    unit_costs: float

class TaxCalculator:
    """Handles tax calculations and incentives."""
    
    def __init__(self,
                 federal_rate: float,
                 state_rate: float,
                 severance_rate: float,
                 depletion_allowance: float,
                 tax_credits: Dict[str, float]):
        """
        Initialize tax calculator.
        
        Args:
            federal_rate: Federal tax rate
            state_rate: State tax rate
            severance_rate: Severance tax rate
            depletion_allowance: Depletion allowance percentage
            tax_credits: Dictionary of available tax credits
        """
        self.federal_rate = federal_rate
        self.state_rate = state_rate
        self.severance_rate = severance_rate
        self.depletion_allowance = depletion_allowance
        self.tax_credits = tax_credits
        
        self.tax_history: List[Dict] = []
    
    def calculate_depletion_allowance(self, revenue: float) -> float:
        """
        Calculate depletion allowance.
        
        Args:
            revenue: Gross revenue
            
        Returns:
            Depletion allowance amount
        """
        return revenue * self.depletion_allowance
    
    def calculate_severance_tax(self, production: float, price: float) -> float:
        """
        Calculate severance tax.
        
        Args:
            production: Oil production in barrels
            price: Oil price per barrel
            
        Returns:
            Severance tax amount
        """
        return production * price * self.severance_rate
    
    def apply_tax_credits(self, tax_obligation: float) -> float:
        """
        Apply available tax credits.
        
        Args:
            tax_obligation: Initial tax obligation
            
        Returns:
            Adjusted tax obligation
        """
        total_credits = sum(self.tax_credits.values())
        return max(0, tax_obligation - total_credits)
    
    def calculate_obligations(self, revenue: float, costs: float) -> float:
        """
        Calculate total tax obligations.
        
        Args:
            revenue: Gross revenue
            costs: Total costs
            
        Returns:
            Total tax obligation
        """
        taxable_income = revenue - costs
        depletion = self.calculate_depletion_allowance(revenue)
        taxable_income -= depletion
        
        federal_tax = max(0, taxable_income * self.federal_rate)
        state_tax = max(0, taxable_income * self.state_rate)
        severance_tax = self.calculate_severance_tax(revenue / 70, 70)  # Assuming $70/bbl
        
        total_tax = federal_tax + state_tax + severance_tax
        adjusted_tax = self.apply_tax_credits(total_tax)
        
        self.tax_history.append({
            'date': datetime.now(),
            'revenue': revenue,
            'costs': costs,
            'depletion': depletion,
            'federal_tax': federal_tax,
            'state_tax': state_tax,
            'severance_tax': severance_tax,
            'credits_applied': total_tax - adjusted_tax,
            'final_obligation': adjusted_tax
        })
        
        return adjusted_tax