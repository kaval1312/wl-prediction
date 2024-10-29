from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.stats import weibull_min

@dataclass
class EquipmentFailure:
    """Represents an equipment failure event."""
    component_name: str
    time: int
    cost: float
    downtime: int
    impact_severity: float
    repair_time: int

class EquipmentComponent:
    """Represents an oil well equipment component with failure modeling."""
    
    def __init__(self, 
                 name: str,
                 expected_life: int,
                 replacement_cost: float,
                 failure_impact: float,
                 maintenance_schedule: int,
                 operating_conditions: float = 1.0,
                 installation_date: Optional[str] = None):
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
        """
        self.name = name
        self.expected_life = expected_life
        self.replacement_cost = replacement_cost
        self.failure_impact = failure_impact
        self.maintenance_schedule = maintenance_schedule
        self.operating_conditions = operating_conditions
        self.installation_date = installation_date
        
        self.maintenance_history: List[Dict] = []
        self.failure_history: List[EquipmentFailure] = []
        
        # Weibull distribution parameters
        self.shape_parameter = 2.5  # Shape parameter for wear-out failures
        self.scale_parameter = self._calculate_scale_parameter()
    
    def _calculate_scale_parameter(self) -> float:
        """Calculate Weibull scale parameter based on operating conditions."""
        base_scale = self.expected_life / np.log(2) ** (1/self.shape_parameter)
        condition_factor = 0.7 + (0.3 * self.operating_conditions)
        return base_scale * condition_factor
    
    def calculate_failure_probability(self, months: int) -> np.ndarray:
        """
        Calculate probability of failure over time.
        
        Args:
            months: Number of months to calculate
            
        Returns:
            Array of failure probabilities
        """
        time_points = np.arange(months)
        failure_prob = weibull_min.cdf(
            time_points,
            self.shape_parameter,
            loc=0,
            scale=self.scale_parameter
        )
        
        # Adjust for maintenance history
        maintenance_effect = len(self.maintenance_history) * 0.1
        failure_prob = failure_prob * (1 - maintenance_effect)
        
        return failure_prob
    
    def simulate_failures(self, months: int) -> List[EquipmentFailure]:
        """
        Simulate equipment failures over time.
        
        Args:
            months: Number of months to simulate
            
        Returns:
            List of simulated failures
        """
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
                repair_time=int(5 + 10 * np.random.random())
            )
            simulated_failures.append(failure)
        
        return simulated_failures
    
    def schedule_maintenance(self) -> List[Dict]:
        """
        Schedule preventive maintenance activities.
        
        Returns:
            List of scheduled maintenance activities
        """
        months_ahead = self.expected_life
        maintenance_schedule = []
        
        current_month = 0
        while current_month < months_ahead:
            maintenance_schedule.append({
                'month': current_month,
                'type': 'preventive',
                'cost': self.replacement_cost * 0.2,
                'duration': 2
            })
            current_month += self.maintenance_schedule
        
        return maintenance_schedule
    
    def calculate_reliability(self, time: int) -> float:
        """
        Calculate equipment reliability at given time.
        
        Args:
            time: Time in months
            
        Returns:
            Reliability score (0-1)
        """
        return 1 - self.calculate_failure_probability(time + 1)[time]
    
    def estimate_remaining_life(self) -> float:
        """
        Estimate remaining useful life.
        
        Returns:
            Estimated remaining life in months
        """
        current_reliability = self.calculate_reliability(0)
        return self.expected_life * current_reliability * self.operating_conditions
