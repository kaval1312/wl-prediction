import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class EquipmentAnalyzer:
    def __init__(self):
        self.failure_modes = {
            'pump': ['cavitation', 'bearing_wear', 'seal_failure'],
            'motor': ['overheating', 'electrical_failure', 'bearing_wear'],
            'separator': ['corrosion', 'level_control', 'pressure_control']
        }
        
        self.thresholds = {
            'pump': {
                'vibration': 0.5,
                'temperature': 180,
                'pressure': 1000
            },
            'motor': {
                'temperature': 200,
                'current': 100,
                'voltage': 480
            },
            'separator': {
                'pressure': 300,
                'level': 80,
                'temperature': 150
            }
        }

    def calculate_reliability(
        self,
        equipment_type: str,
        parameters: Dict[str, float],
        age: float,
        maintenance_history: List[datetime]
    ) -> Tuple[float, List[str]]:
        """
        Calculate equipment reliability and identify risks
        """
        # Base reliability based on age
        base_reliability = np.exp(-0.1 * age)
        
        # Maintenance factor
        maintenance_factor = self._calculate_maintenance_factor(maintenance_history)
        
        # Parameter violations
        violations = []
        violation_factor = 0
        
        for param, value in parameters.items():
            if param in self.thresholds[equipment_type]:
                threshold = self.thresholds[equipment_type][param]
                if value > threshold:
                    violations.append(f"{param} exceeds threshold")
                    violation_factor += 0.1
        
        # Calculate final reliability
        reliability = base_reliability * (1 + maintenance_factor) * (1 - violation_factor)
        reliability = max(0, min(1, reliability))
        
        return reliability, violations

    def _calculate_maintenance_factor(
        self,
        maintenance_history: List[datetime]
    ) -> float:
        """
        Calculate maintenance effectiveness factor
        """
        if not maintenance_history:
            return 0
            
        now = datetime.now()
        recent_maintenance = [
            m for m in maintenance_history
            if (now - m).days < 180
        ]
        
        return len(recent_maintenance) * 0.05

    def predict_failure(
        self,
        reliability: float,
        parameters: Dict[str, float],
        threshold: float = 0.7
    ) -> Tuple[bool, int]:
        """
        Predict equipment failure and estimate days until maintenance needed
        """
        risk_level = 1 - reliability
        needs_maintenance = risk_level > threshold
        
        if needs_maintenance:
            days_until_critical = int((threshold - risk_level) / 0.01)
            days_until_critical = max(0, days_until_critical)
        else:
            days_until_critical = int((threshold - risk_level) / 0.01)
        
        return needs_maintenance, days_until_critical