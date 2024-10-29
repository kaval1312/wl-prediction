# src/utils/lease_terms.py

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Union


@dataclass
class LeaseTerms:
    """Well lease terms and conditions"""
    working_interest: float  # Your working interest percentage
    net_revenue_interest: float  # Your net revenue interest
    royalty_rate: float  # Royalty rate to mineral rights owners
    lease_bonus: float  # Initial lease bonus payment
    lease_term_years: int  # Primary lease term in years
    extension_cost: float  # Cost to extend lease
    minimum_royalty: float  # Minimum royalty payment required


@dataclass
class AbandonmentCosts:
    """Well abandonment cost structure"""
    plugging_cost: float  # Cost to plug the well
    site_restoration: float  # Cost to restore the site
    equipment_removal: float  # Cost to remove equipment
    environmental_cleanup: float  # Environmental cleanup costs
    regulatory_fees: float  # Regulatory filing fees
    contingency: float  # Contingency percentage


class LeaseAnalyzer:
    """Analyzer for lease economics and abandonment obligations"""

    def __init__(self):
        self.inflation_rate = 0.03  # Annual inflation rate

    def calculate_lease_economics(
            self,
            production: np.ndarray,
            oil_price: float,
            lease_terms: LeaseTerms,
            current_month: int
    ) -> Dict[str, float]:
        """Calculate lease economics"""
        # Calculate gross revenue
        gross_revenue = production * oil_price

        # Calculate royalties
        royalties = gross_revenue * lease_terms.royalty_rate
        minimum_royalty_obligation = np.full_like(production, lease_terms.minimum_royalty)
        actual_royalties = np.maximum(royalties, minimum_royalty_obligation)

        # Calculate working interest revenue
        working_interest_revenue = (gross_revenue - actual_royalties) * lease_terms.working_interest

        # Calculate net revenue
        net_revenue = working_interest_revenue * (lease_terms.net_revenue_interest / lease_terms.working_interest)

        # Calculate lease obligations
        months_remaining = lease_terms.lease_term_years * 12 - current_month
        extension_needed = months_remaining < 0

        return {
            'gross_revenue': float(np.sum(gross_revenue)),
            'total_royalties': float(np.sum(actual_royalties)),
            'working_interest_revenue': float(np.sum(working_interest_revenue)),
            'net_revenue': float(np.sum(net_revenue)),
            'extension_needed': extension_needed,
            'extension_cost': lease_terms.extension_cost if extension_needed else 0,
            'remaining_term_months': max(0, months_remaining)
        }

    def calculate_abandonment_costs(
            self,
            well_depth: float,
            well_age: float,
            equipment_count: int,
            environmental_risk: str = 'low'
    ) -> AbandonmentCosts:
        """Calculate abandonment costs"""
        # Base plugging cost
        plugging_cost = well_depth * 25  # $25 per foot

        # Adjust for well age
        age_factor = 1 + (well_age / 20)  # 5% increase per year of age
        plugging_cost *= age_factor

        # Equipment removal
        equipment_removal = equipment_count * 5000  # $5000 per piece

        # Environmental costs based on risk
        environmental_costs = {
            'low': 25000,
            'medium': 75000,
            'high': 150000
        }

        # Site restoration
        site_restoration = well_depth * 5  # $5 per foot

        return AbandonmentCosts(
            plugging_cost=plugging_cost,
            site_restoration=site_restoration,
            equipment_removal=equipment_removal,
            environmental_cleanup=environmental_costs[environmental_risk],
            regulatory_fees=5000,
            contingency=0.15
        )

    def calculate_total_obligation(
            self,
            costs: AbandonmentCosts,
            inflation_years: float
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Calculate total abandonment obligation with inflation"""
        base_total = (
                costs.plugging_cost +
                costs.site_restoration +
                costs.equipment_removal +
                costs.environmental_cleanup +
                costs.regulatory_fees
        )

        subtotal = base_total * (1 + costs.contingency)
        inflation_factor = (1 + self.inflation_rate) ** inflation_years
        total_with_inflation = subtotal * inflation_factor

        return {
            'base_costs': base_total,
            'contingency_amount': base_total * costs.contingency,
            'subtotal': subtotal,
            'inflation_adjustment': total_with_inflation - subtotal,
            'total_obligation': total_with_inflation,
            'cost_breakdown': {
                'Plugging': costs.plugging_cost,
                'Site Restoration': costs.site_restoration,
                'Equipment Removal': costs.equipment_removal,
                'Environmental': costs.environmental_cleanup,
                'Regulatory': costs.regulatory_fees
            }
        }