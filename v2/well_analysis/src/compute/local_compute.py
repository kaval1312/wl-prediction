import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
import logging
from datetime import datetime
from pathlib import Path
import json
import traceback
from ..core.data_structures import WellData
from ..analysis.monte_carlo import MonteCarloSimulator
from ..analysis.decline_analysis import DeclineAnalyzer
from ..analysis.economic_analysis import EconomicAnalyzer
from ..analysis.environmental_analysis import EnvironmentalAnalyzer
from ..analysis.maintenance_analysis import MaintenanceAnalyzer

logger = logging.getLogger(__name__)


class LocalCompute:
    def __init__(self, n_processes: int = None):
        """Initialize local compute manager"""
        self.n_processes = n_processes if n_processes else mp.cpu_count()
        self.results_dir = Path('results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _run_single_analysis(self, analysis_func: Callable,
                             analysis_params: Dict) -> Dict:
        """Run a single analysis task"""
        try:
            return analysis_func(**analysis_params)
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def run_parallel_analysis(self, analysis_func: Callable,
                              param_list: List[Dict]) -> List[Dict]:
        """Run multiple analyses in parallel"""
        try:
            with mp.Pool(self.n_processes) as pool:
                results = pool.starmap(
                    self._run_single_analysis,
                    [(analysis_func, params) for params in param_list]
                )
            return results
        except Exception as e:
            logger.error(f"Error in parallel analysis: {str(e)}")
            return [{}] * len(param_list)

    def run_monte_carlo(self, well_data: WellData,
                        n_simulations: int = 1000,
                        n_processes: int = None) -> Dict:
        """Run Monte Carlo simulation locally"""
        try:
            simulator = MonteCarloSimulator()

            # Split simulations across processes
            n_proc = n_processes if n_processes else self.n_processes
            sims_per_process = n_simulations // n_proc

            param_list = [
                {
                    'well_data': well_data,
                    'n_simulations': sims_per_process
                }
                for _ in range(n_proc)
            ]

            # Run simulations in parallel
            results = self.run_parallel_analysis(
                simulator.run_integrated_simulation,
                param_list
            )

            # Combine results
            combined_results = self._combine_monte_carlo_results(results)

            # Save results
            self._save_results(combined_results, 'monte_carlo')

            return combined_results

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}

    def _combine_monte_carlo_results(self, results: List[Dict]) -> Dict:
        """Combine results from parallel Monte Carlo simulations"""
        try:
            combined = {}
            for key in ['production', 'economics', 'maintenance', 'environmental']:
                combined[key] = {}
                for subkey in results[0][key].keys():
                    if isinstance(results[0][key][subkey], np.ndarray):
                        combined[key][subkey] = np.concatenate(
                            [r[key][subkey] for r in results]
                        )
                    elif isinstance(results[0][key][subkey], dict):
                        combined[key][subkey] = {}
                        for metric in results[0][key][subkey].keys():
                            combined[key][subkey][metric] = np.concatenate(
                                [r[key][subkey][metric] for r in results]
                            )
            return combined
        except Exception as e:
            logger.error(f"Error combining Monte Carlo results: {str(e)}")
            return {}

    def run_decline_analysis(self, well_data: WellData,
                             analysis_params: Dict) -> Dict:
        """Run decline curve analysis locally"""
        try:
            analyzer = DeclineAnalyzer()
            results = analyzer.analyze_well_decline(well_data)
            self._save_results(results, 'decline')
            return results
        except Exception as e:
            logger.error(f"Error in decline analysis: {str(e)}")
            return {}

    def run_economic_analysis(self, well_data: WellData,
                              analysis_params: Dict) -> Dict:
        """Run economic analysis locally"""
        try:
            analyzer = EconomicAnalyzer()
            results = analyzer.analyze_economics(
                well_data,
                analysis_params.get('oil_price', 70),
                analysis_params.get('gas_price', 3.5)
            )
            self._save_results(results, 'economic')
            return results
        except Exception as e:
            logger.error(f"Error in economic analysis: {str(e)}")
            return {}

    def run_environmental_analysis(self, well_data: WellData,
                                   analysis_params: Dict) -> Dict:
        """Run environmental analysis locally"""
        try:
            analyzer = EnvironmentalAnalyzer()
            results = analyzer.generate_environmental_report(well_data)
            self._save_results(results, 'environmental')
            return results
        except Exception as e:
            logger.error(f"Error in environmental analysis: {str(e)}")
            return {}

    def run_maintenance_analysis(self, well_data: WellData,
                                 analysis_params: Dict) -> Dict:
        """Run maintenance analysis locally"""
        try:
            analyzer = MaintenanceAnalyzer()
            results = analyzer.generate_maintenance_report(well_data)
            self._save_results(results, 'maintenance')
            return results
        except Exception as e:
            logger.error(f"Error in maintenance analysis: {str(e)}")
            return {}

    def run_integrated_analysis(self, well_data: WellData,
                                analysis_params: Dict) -> Dict:
        """Run all analyses locally"""
        try:
            results = {
                'monte_carlo': self.run_monte_carlo(
                    well_data,
                    analysis_params.get('n_simulations', 1000)
                ),
                'decline': self.run_decline_analysis(
                    well_data,
                    analysis_params
                ),
                'economic': self.run_economic_analysis(
                    well_data,
                    analysis_params
                ),
                'environmental': self.run_environmental_analysis(
                    well_data,
                    analysis_params
                ),
                'maintenance': self.run_maintenance_analysis(
                    well_data,
                    analysis_params
                )
            }

            self._save_results(results, 'integrated')
            return results

        except Exception as e:
            logger.error(f"Error in integrated analysis: {str(e)}")
            return {}

    def _save_results(self, results: Dict, analysis_type: str) -> None:
        """Save analysis results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.results_dir / f"{analysis_type}_{timestamp}.json"

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            serializable_results = convert_numpy(results)

            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def load_results(self, analysis_type: str,
                     timestamp: str = None) -> Optional[Dict]:
        """Load analysis results from file"""
        try:
            if timestamp:
                filename = self.results_dir / f"{analysis_type}_{timestamp}.json"
            else:
                # Get most recent results
                files = list(self.results_dir.glob(f"{analysis_type}_*.json"))
                if not files:
                    return None
                filename = max(files, key=lambda x: x.stat().st_mtime)

            with open(filename, 'r') as f:
                results = json.load(f)

            return results

        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return None

    def get_available_results(self) -> Dict[str, List[str]]:
        """Get list of available results by analysis type"""
        try:
            results = {}
            for file in self.results_dir.glob("*.json"):
                analysis_type = file.stem.split('_')[0]
                timestamp = '_'.join(file.stem.split('_')[1:])

                if analysis_type not in results:
                    results[analysis_type] = []
                results[analysis_type].append(timestamp)

            return results

        except Exception as e:
            logger.error(f"Error getting available results: {str(e)}")
            return {}

    def cleanup_old_results(self, max_age_days: int = 30) -> None:
        """Clean up old result files"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            for file in self.results_dir.glob("*.json"):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()

        except Exception as e:
            logger.error(f"Error cleaning up old results: {str(e)}")
