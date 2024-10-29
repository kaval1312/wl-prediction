# src/utils/equipment.py
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime


class EquipmentAnalyzer:
    """Analyzer for equipment health and maintenance"""

    def __init__(self):
        self.thresholds = {
            'vibration': 0.5,  # Maximum acceptable vibration
            'temperature': 180,  # Maximum acceptable temperature (°F)
            'pressure': 1000  # Maximum acceptable pressure (psi)
        }

        self.maintenance_intervals = {
            'routine': 30,  # days
            'major': 180,  # days
            'inspection': 90  # days
        }

    def calculate_reliability(
            self,
            vibration: float,
            temperature: float,
            age_factor: float = 1.0
    ) -> float:
        """
        Calculate equipment reliability based on operating parameters.

        Args:
            vibration: Current vibration level
            temperature: Current temperature
            age_factor: Equipment age factor (1.0 = new)

        Returns:
            Reliability score (0-1)
        """
        # Normalize parameters to 0-1 scale
        vib_score = max(0, 1 - (vibration / self.thresholds['vibration']))
        temp_score = max(0, 1 - (temperature / self.thresholds['temperature']))

        # Weight the factors
        reliability = (0.4 * vib_score + 0.4 * temp_score + 0.2 * age_factor)

        return max(0, min(1, reliability))

    def predict_failure(
            self,
            vibration: float,
            temperature: float,
            health: float
    ) -> float:
        """
        Predict probability of equipment failure.

        Args:
            vibration: Current vibration level
            temperature: Current temperature
            health: Current health score (0-100)

        Returns:
            Failure probability (0-1)
        """
        # Normalize inputs
        vib_factor = min(1.0, vibration / self.thresholds['vibration'])
        temp_factor = max(0.0, (temperature - self.thresholds['temperature']) / 100)
        health_factor = 1.0 - (health / 100)

        # Calculate failure probability
        failure_prob = (0.4 * vib_factor + 0.3 * temp_factor + 0.3 * health_factor)
        return min(1.0, max(0.0, failure_prob))

    def analyze_maintenance_needs(
            self,
            equipment_data: Dict[str, float],
            last_maintenance: datetime
    ) -> Dict[str, any]:
        """
        Analyze maintenance requirements

        Args:
            equipment_data: Dictionary of equipment parameters
            last_maintenance: Date of last maintenance

        Returns:
            Dictionary of maintenance analysis
        """
        days_since_maintenance = (datetime.now() - last_maintenance).days

        reliability = self.calculate_reliability(
            vibration=equipment_data.get('vibration', 0),
            temperature=equipment_data.get('temperature', 0)
        )

        failure_prob = self.predict_failure(
            vibration=equipment_data.get('vibration', 0),
            temperature=equipment_data.get('temperature', 0),
            health=reliability * 100
        )

        return {
            'reliability': reliability,
            'failure_probability': failure_prob,
            'days_since_maintenance': days_since_maintenance,
            'maintenance_due': days_since_maintenance > self.maintenance_intervals['routine'],
            'inspection_due': days_since_maintenance > self.maintenance_intervals['inspection'],
            'major_service_due': days_since_maintenance > self.maintenance_intervals['major']
        }

    def get_maintenance_recommendations(
            self,
            analysis_results: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Generate maintenance recommendations

        Args:
            analysis_results: Results from analyze_maintenance_needs

        Returns:
            List of maintenance recommendations
        """
        recommendations = []

        if analysis_results['failure_probability'] > 0.7:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Immediate maintenance required',
                'timeframe': 'Within 24 hours',
                'risk': 'Critical failure risk'
            })
        elif analysis_results['failure_probability'] > 0.4:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Schedule maintenance',
                'timeframe': 'Within 7 days',
                'risk': 'Increased failure risk'
            })

        if analysis_results['maintenance_due']:
            recommendations.append({
                'priority': 'NORMAL',
                'action': 'Routine maintenance',
                'timeframe': 'Schedule next available',
                'risk': 'Regular maintenance interval exceeded'
            })

        return recommendations