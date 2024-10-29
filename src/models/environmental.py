from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta


@dataclass
class ComplianceViolation:
    """Represents an environmental compliance violation."""
    regulation_name: str
    violation_date: datetime
    severity: float  # 0-1 scale
    penalty: float
    description: str
    remediation_cost: float
    remediation_time: int  # days
    resolved: bool = False
    resolution_date: Optional[datetime] = None
    remediation_plan: Optional[str] = None
    inspector_notes: Optional[str] = None


class EnvironmentalRegulation:
    """Represents environmental regulations and compliance requirements."""

    def __init__(
            self,
            name: str,
            compliance_cost: float = 5000.0,
            recurring_cost: float = 500.0,
            penalty: float = 25000.0,
            inspection_frequency: int = 3,
            probability_of_violation: float = 0.05,
            regulatory_body: Optional[str] = None
    ):
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
        self.last_inspection_date: Optional[datetime] = None
        self.next_inspection_date: Optional[datetime] = None

    def calculate_monthly_cost(self, months: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculate monthly compliance costs.

        Args:
            months: Number of months to calculate

        Returns:
            Tuple of (total_costs, cost_components)
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

        total_costs = monthly_costs + inspections + penalties

        cost_components = {
            'recurring': monthly_costs,
            'inspections': inspections,
            'penalties': penalties
        }

        return total_costs, cost_components

    def simulate_violations(
            self,
            months: int,
            random_seed: Optional[int] = None
    ) -> List[ComplianceViolation]:
        """
        Simulate compliance violations.

        Args:
            months: Number of months to simulate
            random_seed: Random seed for reproducibility

        Returns:
            List of simulated violations
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        violations = []
        start_date = datetime.now()

        for month in range(months):
            if np.random.random() < self.probability_of_violation:
                violation_date = start_date + timedelta(days=month * 30)
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

    def schedule_inspections(
            self,
            start_date: datetime,
            months: int
    ) -> List[Dict]:
        """
        Schedule regulatory inspections.

        Args:
            start_date: Start date for scheduling
            months: Number of months to schedule

        Returns:
            List of scheduled inspections
        """
        inspections = []

        for month in range(0, months, self.inspection_frequency):
            inspection_date = start_date + timedelta(days=month * 30)
            inspections.append({
                'date': inspection_date,
                'cost': self.compliance_cost,
                'type': 'routine',
                'duration': 2,
                'regulation': self.name,
                'status': 'scheduled'
            })

        if inspections:
            self.next_inspection_date = inspections[0]['date']

        return inspections

    def record_inspection(
            self,
            date: datetime,
            result: str,
            inspector: str,
            findings: List[str],
            cost: Optional[float] = None
    ) -> None:
        """
        Record an inspection result.

        Args:
            date: Inspection date
            result: Inspection result (pass/fail)
            inspector: Inspector name
            findings: List of inspection findings
            cost: Actual inspection cost
        """
        inspection = {
            'date': date,
            'result': result,
            'inspector': inspector,
            'findings': findings,
            'cost': cost or self.compliance_cost,
            'regulation': self.name
        }

        self.inspection_history.append(inspection)
        self.last_inspection_date = date

        # Schedule next inspection
        self.next_inspection_date = date + timedelta(days=self.inspection_frequency * 30)

    def record_violation(
            self,
            violation: ComplianceViolation
    ) -> None:
        """
        Record a compliance violation.

        Args:
            violation: ComplianceViolation instance
        """
        self.violation_history.append(violation)

    def resolve_violation(
            self,
            violation: ComplianceViolation,
            resolution_date: datetime,
            resolution_notes: str
    ) -> None:
        """
        Mark a violation as resolved.

        Args:
            violation: ComplianceViolation to resolve
            resolution_date: Date of resolution
            resolution_notes: Notes about resolution
        """
        violation.resolved = True
        violation.resolution_date = resolution_date
        violation.inspector_notes = resolution_notes

    def calculate_compliance_score(self) -> float:
        """
        Calculate current compliance score.

        Returns:
            Compliance score (0-1)
        """
        if not self.violation_history:
            return 1.0

        # Consider only violations in the last year
        recent_violations = [
            v for v in self.violation_history
            if (datetime.now() - v.violation_date).days < 365
        ]

        if not recent_violations:
            return 0.9

        # Calculate score based on violation severity and resolution
        total_impact = 0
        for violation in recent_violations:
            impact = violation.severity
            if violation.resolved:
                impact *= 0.5  # Reduce impact of resolved violations
            total_impact += impact

        average_impact = total_impact / len(recent_violations)
        return max(0, 1 - average_impact)

    def get_compliance_status(self) -> Dict[str, any]:
        """
        Get current compliance status summary.

        Returns:
            Dictionary with compliance status information
        """
        compliance_score = self.calculate_compliance_score()

        return {
            'regulation_name': self.name,
            'compliance_score': compliance_score,
            'last_inspection': self.last_inspection_date,
            'next_inspection': self.next_inspection_date,
            'open_violations': len([v for v in self.violation_history if not v.resolved]),
            'total_violations': len(self.violation_history),
            'recent_violations': len([
                v for v in self.violation_history
                if (datetime.now() - v.violation_date).days < 90
            ]),
            'status': 'Compliant' if compliance_score >= 0.7 else 'At Risk',
            'regulatory_body': self.regulatory_body
        }

    def estimate_annual_cost(self) -> Dict[str, float]:
        """
        Estimate annual compliance costs.

        Returns:
            Dictionary of estimated annual costs
        """
        annual_inspections = 12 / self.inspection_frequency
        estimated_violations = 12 * self.probability_of_violation

        costs = {
            'recurring_costs': self.recurring_cost * 12,
            'inspection_costs': self.compliance_cost * annual_inspections,
            'estimated_penalties': self.penalty * estimated_violations,
            'total_estimated_cost': (self.recurring_cost * 12 +
                                     self.compliance_cost * annual_inspections +
                                     self.penalty * estimated_violations)
        }

        return costs