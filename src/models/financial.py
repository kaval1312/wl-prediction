from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime


@dataclass
class FinancialMetrics:
    """Container for financial metrics."""
    npv: float
    irr: Optional[float]
    payback_period: float
    roi: float
    break_even_price: float
    total_revenue: float
    total_costs: float
    unit_costs: float
    profit_margin: float

    @property
    def net_profit(self) -> float:
        return self.total_revenue - self.total_costs


class TaxCalculator:
    """Handles tax calculations and incentives."""

    def __init__(
            self,
            federal_rate: float = 0.21,
            state_rate: float = 0.05,
            severance_rate: float = 0.045,
            depletion_allowance: float = 0.15,
            tax_credits: Dict[str, float] = None
    ):
        """
        Initialize tax calculator.

        Args:
            federal_rate: Federal corporate tax rate
            state_rate: State corporate tax rate
            severance_rate: Oil severance tax rate
            depletion_allowance: Percentage depletion allowance
            tax_credits: Dictionary of available tax credits
        """
        self.federal_rate = federal_rate
        self.state_rate = state_rate
        self.severance_rate = severance_rate
        self.depletion_allowance = depletion_allowance
        self.tax_credits = tax_credits or {}

        self.tax_history: List[Dict] = []

    def calculate_depletion_allowance(self, revenue: float) -> float:
        """Calculate depletion allowance."""
        return revenue * self.depletion_allowance

    def calculate_severance_tax(
            self,
            production: float,
            price: float,
            exemptions: Dict[str, float] = None
    ) -> float:
        """
        Calculate severance tax.

        Args:
            production: Oil production in barrels
            price: Oil price per barrel
            exemptions: Dictionary of tax exemptions
        """
        taxable_value = production * price

        if exemptions:
            for exemption_rate in exemptions.values():
                taxable_value *= (1 - exemption_rate)

        return taxable_value * self.severance_rate

    def apply_tax_credits(self, tax_obligation: float) -> float:
        """Apply available tax credits."""
        total_credits = sum(self.tax_credits.values())
        return max(0, tax_obligation - total_credits)

    def calculate_monthly_obligations(
            self,
            monthly_revenue: float,
            monthly_costs: float,
            production: float,
            oil_price: float
    ) -> Dict[str, float]:
        """
        Calculate monthly tax obligations.

        Args:
            monthly_revenue: Monthly gross revenue
            monthly_costs: Monthly total costs
            production: Monthly oil production
            oil_price: Oil price per barrel
        """
        # Calculate taxable income
        depletion = self.calculate_depletion_allowance(monthly_revenue)
        taxable_income = monthly_revenue - monthly_costs - depletion

        # Calculate tax components
        federal_tax = max(0, taxable_income * self.federal_rate)
        state_tax = max(0, taxable_income * self.state_rate)
        severance_tax = self.calculate_severance_tax(production, oil_price)

        # Apply credits
        total_tax = federal_tax + state_tax + severance_tax
        final_tax = self.apply_tax_credits(total_tax)

        tax_components = {
            'federal_tax': federal_tax,
            'state_tax': state_tax,
            'severance_tax': severance_tax,
            'depletion': depletion,
            'credits_applied': total_tax - final_tax,
            'final_obligation': final_tax
        }

        # Record tax history
        self.tax_history.append({
            'date': datetime.now(),
            'revenue': monthly_revenue,
            'costs': monthly_costs,
            'production': production,
            'oil_price': oil_price,
            **tax_components
        })

        return tax_components

    def get_tax_summary(self) -> Dict[str, float]:
        """Get summary of tax history."""
        if not self.tax_history:
            return {}

        return {
            'total_federal_tax': sum(h['federal_tax'] for h in self.tax_history),
            'total_state_tax': sum(h['state_tax'] for h in self.tax_history),
            'total_severance_tax': sum(h['severance_tax'] for h in self.tax_history),
            'total_depletion': sum(h['depletion'] for h in self.tax_history),
            'total_credits_applied': sum(h['credits_applied'] for h in self.tax_history),
            'total_tax_paid': sum(h['final_obligation'] for h in self.tax_history),
            'effective_tax_rate': sum(h['final_obligation'] for h in self.tax_history) /
                                  sum(h['revenue'] for h in self.tax_history)
            if sum(h['revenue'] for h in self.tax_history) > 0 else 0
        }