from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass
class ComplianceViolation:
    """Represents an environmental compliance violation."""
    regulation_name: str
    violation_date: datetime
    severity: float
    penalty: float
    description: str
    remediation_cost: float
    remediation_time: int

class EnvironmentalRegulation:
    """Represents environmental regulations and compliance requirements."""
    
    def __init__(self,
                 name: str,
                 compliance_cost: float,
                 recurring_cost: float,
                 penalty: float,
                 inspection_frequency: int,
                 probability_of_violation: float,
                 regulatory_body: Optional[str] = None):
        """
        Initialize environmental regulation.
        
        Args:
            name: Regulation identifier
            compliance_cost: Initial compliance cost
            recurring_cost: Monthly compliance cost
            penalty: Violation penalty cost
            inspection_frequency: Months between inspections
            probability_of_violation: Monthly violation probability
            regulatory_body: Name of regulatory authority
        """
        self.name = name
        self.compliance_cost = compliance_cost
        self.recurring_cost = recurring_cost
        self.penalty = penalty
        self.inspection_frequency = inspection_frequency
        self.probability_of_violation = probability_of_violation
        self.regulatory_body = regulatory_body
        
        self.inspection_history: List[Dict] = []
        self.violation_history: List[ComplianceViolation] = []
    
    def calculate_monthly_cost(self, months: int) -> np.ndarray:
        """
        Calculate monthly compliance costs.
        
        Args:
            months: Number of months to calculate
            
        Returns:
            Array of monthly costs
        """
        # Base recurring costs
        monthly_costs = np.full(months, self.recurring_cost)
        
        # Add inspection costs
        inspections = np.zeros(months)
        inspection_months = np.arange(0, months, self.inspection_frequency)
        inspections[inspection_months] = self.compliance_cost
        
        # Add violations and penalties
        violations = np.random.binomial(1, self.probability_of_violation, months)
        penalties = violations * self.penalty
        
        return monthly_costs + inspections + penalties
    
    def simulate_violations(self, months: int) -> List[ComplianceViolation]:
        """
        Simulate compliance violations.
        
        Args:
            months: Number of months to simulate
            
        Returns:
            List of simulated violations
        """
        violations = []
        start_date = datetime.now()
        
        for month in range(months):
            if np.random.random() < self.probability_of_violation:
                violation_date = start_date + timedelta(days=month*30)
                severity = np.random.triangular(0.1, 0.3, 1.0)
                
                violation = ComplianceViolation(
                    regulation_name=self.name,
                    violation_date=violation_date,
                    severity=severity,
                    penalty=self.penalty * severity,
                    description=f"Violation of {self.name} regulations",
                    remediation_cost=self.compliance_cost * severity,
                    remediation_time=int(30 * severity)
                )
                violations.append(violation)
        
        return violations
    
    def schedule_inspections(self, months: int) -> List[Dict]:
        """
        Schedule regulatory inspections.
        
        Args:
            months: Number of months to schedule
            
        Returns:
            List of scheduled inspections
        """
        inspections = []
        start_date = datetime.now()
        
        for month in range(0, months, self.inspection_frequency):
            inspection_date = start_date + timedelta(days=month*30)
            inspections.append({
                'date': inspection_date,
                'cost': self.compliance_cost,
                'type': 'routine',
                'duration': 2
            })
        
        return inspections
    
    def calculate_compliance_score(self) -> float:
        """
        Calculate current compliance score.
        
        Returns:
            Compliance score (0-1)
        """
        if not self.violation_history:
            return 1.0
            
        recent_violations = [v for v in self.violation_history 
                           if (datetime.now() - v.violation_date).days < 365]
        
        if not recent_violations:
            return 0.9
            
        violation_impact = sum(v.severity for v in recent_violations) / len(recent_violations)
        return max(0, 1 - violation_impact)
