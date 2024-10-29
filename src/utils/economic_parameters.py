# src/utils/economic_parameters.py
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class EconomicParameters:
    """Container for economic calculation parameters"""
    VERSION = "1.2"  # Update version tracking

    def __init__(self, oil_price: float, opex: float, initial_investment: float, discount_rate: float,
                 initial_rate: float, decline_rate: float, working_interest: float, net_revenue_interest: float,
                 lease_terms: float, abandonment_costs: float):
        self.oil_price = oil_price
        self.opex = opex
        self.initial_investment = initial_investment
        self.discount_rate = discount_rate
        self.initial_rate = initial_rate
        self.decline_rate = decline_rate
        self.working_interest = working_interest
        self.net_revenue_interest = net_revenue_interest
        self.lease_terms = lease_terms
        self.abandonment_costs = abandonment_costs
        logger.debug(f"EconomicParameters v{self.VERSION} instance created with fields: {self.__dict__}")

    @classmethod
    def from_dict(cls, params: Dict[str, float]):
        return cls(**params)

logger.debug(f"EconomicParameters v{EconomicParameters.VERSION} class defined with fields: {EconomicParameters.__annotations__}")