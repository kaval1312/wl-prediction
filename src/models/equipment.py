from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.stats import weibull_min
from datetime import datetime, timedelta


@dataclass
class EquipmentFailure:
    """Represents an equipment failure event."""
    component_name: str
    time: int  # Time of failure in months
    cost: float  # Cost of repair/replacement
    downtime: int  # Downtime in days
    impact_severity: float  # Impact severity (0-1)
    repair_time: int  # Time to repair in days
    description: Optional[str] = None


class EquipmentComponent:
    """Represents an oil well equipment component with failure modeling."""

    def __init__(
            self,
            name: str,
            expected_life: int = 60,
            replacement_cost: float = 50000,
            failure_impact: float = 0.8,
            maintenance_schedule: int = 3,
            operating_conditions: float = 1.0,
            installation_date: Optional[str] = None,
            maintenance_cost_factor: float = 0.2,
            criticality: float = 1.0
    ):
        """
        Initialize equipment component.

        Args:
            name: Component identifier
            expected_life: Expected life in months
            replacement_cost: Cost to replace component
            failure_impact: Impact of failure (0-1)
            maintenance_schedule: Months between maintenance
            operating_conditions: Current operating condition (0-1)
            installation_date: Date of installation
            maintenance_cost_factor: Maintenance cost as fraction of replacement
            criticality: Component criticality factor (0-1)
        """
        self.name = name
        self.expected_life = expected_life
        self.replacement_cost = replacement_cost
        self.failure_impact = failure_impact
        self.maintenance_schedule = maintenance_schedule
        self.operating_conditions = operating_conditions
        self.installation_date = installation_date
        self.maintenance_cost_factor = maintenance_cost_factor
        self.criticality = criticality

        self.maintenance_history: List[Dict] = []
        self.failure_history: List[EquipmentFailure] = []

        # Weibull distribution parameters
        self.shape_parameter = 2.5  # Shape parameter for wear-out failures
        self.scale_parameter = self._calculate_scale_parameter()

    def _calculate_scale_parameter(self) -> float:
        """Calculate Weibull scale parameter based on operating conditions."""
        base_scale = self.expected_life / np.log(2) ** (1 / self.shape_parameter)
        condition_factor = 0.7 + (0.3 * self.operating_conditions)
        return base_scale * condition_factor

    def calculate_failure_probability(
            self,
            months: int,
            include_maintenance: bool = True
    ) -> np.ndarray:
        """
        Calculate probability of failure over time.

        Args:
            months: Number of months to calculate
            include_maintenance: Whether to include maintenance effects
        """
        time_points = np.arange(months)
        failure_prob = weibull_min.cdf(
            time_points,
            self.shape_parameter,
            loc=0,
            scale=self.scale_parameter
        )

        if include_maintenance:
            # Adjust for maintenance history
            maintenance_effect = len(self.maintenance_history) * 0.1
            failure_prob = failure_prob * (1 - maintenance_effect)

        return failure_prob

    def simulate_failures(
            self,
            months: int,
            random_seed: Optional[int] = None
    ) -> List[EquipmentFailure]:
        """
        Simulate equipment failures over time.

        Args:
            months: Number of months to simulate
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        failure_prob = self.calculate_failure_probability(months)
        failures = np.random.binomial(1, failure_prob)

        simulated_failures = []
        for month in np.where(failures == 1)[0]:
            failure = EquipmentFailure(
                component_name=self.name,
                time=month,
                cost=self.replacement_cost * (0.8 + 0.4 * np.random.random()),
                downtime=int(30 * self.failure_impact * np.random.random()),
                impact_severity=self.failure_impact * (0.5 + 0.5 * np.random.random()),
                repair_time=int(5 + 10 * np.random.random()),
                description=f"Simulated failure of {self.name}"
            )
            simulated_failures.append(failure)

        return simulated_failures

    def schedule_maintenance(
            self,
            months_ahead: Optional[int] = None
    ) -> List[Dict]:
        """
        Schedule preventive maintenance activities.

        Args:
            months_ahead: Number of months to schedule (default: expected_life)
        """
        if months_ahead is None:
            months_ahead = self.expected_life

        maintenance_schedule = []
        current_month = 0

        while current_month < months_ahead:
            maintenance_schedule.append({
                'month': current_month,
                'type': 'preventive',
                'cost': self.replacement_cost * self.maintenance_cost_factor,
                'duration': 2,
                'component': self.name
            })
            current_month += self.maintenance_schedule

        return maintenance_schedule

    def calculate_reliability(self, time: int) -> float:
        """Calculate equipment reliability at given time."""
        return 1 - self.calculate_failure_probability(time + 1)[time]

    def estimate_remaining_life(
            self,
            confidence_level: float = 0.9
    ) -> Dict[str, float]:
        """
        Estimate remaining useful life.

        Args:
            confidence_level: Confidence level for estimation
        """
        current_reliability = self.calculate_reliability(0)
        base_estimate = self.expected_life * current_reliability * self.operating_conditions

        # Calculate confidence intervals
        lower_bound = base_estimate * (1 - (1 - confidence_level))
        upper_bound = base_estimate * (1 + (1 - confidence_level))

        return {
            'estimate': base_estimate,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }

    def record_maintenance(
            self,
            date: datetime,
            cost: float,
            description: str,
            maintenance_type: str = 'preventive'
    ) -> None:
        """Record maintenance activity."""
        self.maintenance_history.append({
            'date': date,
            'cost': cost,
            'type': maintenance_type,
            'description': description
        })

        # Update operating conditions based on maintenance
        if maintenance_type == 'preventive':
            self.operating_conditions = min(1.0, self.operating_conditions + 0.1)
        else:  # corrective
            self.operating_conditions = 0.9  # Reset to 90% after repair

    def record_failure(
            self,
            failure: EquipmentFailure
    ) -> None:
        """Record equipment failure."""
        self.failure_history.append(failure)

        # Update operating conditions after failure
        self.operating_conditions = max(0.5, self.operating_conditions - failure.impact_severity)