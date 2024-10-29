import numpy as np
from typing import List, Dict, Tuple

class MonteCarloSimulator:
    def __init__(self, seed: int = None):
        if seed:
            np.random.seed(seed)

    def simulate_production(
        self,
        initial_rate: float,
        decline_rate: float,
        months: int,
        iterations: int = 1000,
        uncertainty: float = 0.1
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation for production forecasting
        """
        results = []
        for _ in range(iterations):
            # Add randomness to parameters
            noise_decline = np.random.normal(decline_rate, decline_rate * uncertainty)
            noise_initial = np.random.normal(initial_rate, initial_rate * uncertainty)
            
            # Calculate production with noise
            time = np.arange(months)
            production = noise_initial * np.exp(-noise_decline * time)
            
            results.append(production)
        
        return np.array(results)

    def get_percentiles(
        self,
        simulations: np.ndarray,
        percentiles: List[float] = [10, 50, 90]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate percentiles from simulation results
        """
        return {
            f"P{p}": np.percentile(simulations, p, axis=0)
            for p in percentiles
        }

    def calculate_statistics(
        self,
        simulations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate statistical measures from simulation results
        """
        return {
            'mean': np.mean(simulations),
            'std': np.std(simulations),
            'min': np.min(simulations),
            'max': np.max(simulations),
            'range': np.ptp(simulations)
        }