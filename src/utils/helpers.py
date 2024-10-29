# src/utils/helpers.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import yaml
import json


class DataValidator:
    """Validate input data and parameters."""

    @staticmethod
    def validate_production_params(params: Dict) -> bool:
        """Validate production parameters."""
        required_fields = ['initial_rate', 'decline_rate', 'forecast_months']

        # Check required fields
        for field in required_fields:
            if field not in params:
                raise ValueError(f"Missing required field: {field}")

        # Validate ranges
        if params['initial_rate'] <= 0:
            raise ValueError("Initial rate must be positive")

        if not 0 <= params['decline_rate'] <= 1:
            raise ValueError("Decline rate must be between 0 and 1")

        if params['forecast_months'] < 1:
            raise ValueError("Forecast months must be positive")

        return True

    @staticmethod
    def validate_economic_params(params: Dict) -> bool:
        """Validate economic parameters."""
        required_fields = ['oil_price', 'opex', 'discount_rate']

        for field in required_fields:
            if field not in params:
                raise ValueError(f"Missing required field: {field}")

        if params['oil_price'] <= 0:
            raise ValueError("Oil price must be positive")

        if params['opex'] < 0:
            raise ValueError("Operating cost cannot be negative")

        if not 0 <= params['discount_rate'] <= 1:
            raise ValueError("Discount rate must be between 0 and 1")

        return True


class DataProcessor:
    """Process and transform data."""

    @staticmethod
    def calculate_decline_curve(
            initial_rate: float,
            decline_rate: float,
            months: int,
            b_factor: float = 0
    ) -> np.ndarray:
        """Calculate production decline curve."""
        time = np.arange(months)
        if b_factor == 0:
            # Exponential decline
            production = initial_rate * np.exp(-decline_rate * time)
        else:
            # Hyperbolic decline
            production = initial_rate / (1 + b_factor * decline_rate * time) ** (1 / b_factor)
        return production

    @staticmethod
    def interpolate_missing(
            df: pd.DataFrame,
            columns: List[str],
            method: str = 'linear'
    ) -> pd.DataFrame:
        """Interpolate missing values in specified columns."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = result[col].interpolate(method=method)
        return result

    @staticmethod
    def calculate_moving_average(
            data: np.ndarray,
            window: int = 3
    ) -> np.ndarray:
        """Calculate moving average of data."""
        return np.convolve(data, np.ones(window) / window, mode='valid')


class FileHandler:
    """Handle file operations."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {str(e)}")

    @staticmethod
    def export_results(
            df: pd.DataFrame,
            results_dir: Union[str, Path],
            prefix: Optional[str] = None
    ) -> Dict[str, Path]:
        """Export analysis results to files."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_" if prefix else ""

        output_files = {}

        # CSV export
        csv_path = results_dir / f"{prefix}results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        output_files['csv'] = csv_path

        # Excel export
        excel_path = results_dir / f"{prefix}results_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # Monthly summary
            monthly_summary = df.resample('M').agg({
                'Production': 'sum',
                'Net_Revenue': 'sum',
                'Water_Cut': 'mean',
                'Equipment_Costs': 'sum'
            })
            monthly_summary.to_excel(writer, sheet_name='Monthly Summary')

            # Financial metrics
            financial_metrics = pd.DataFrame({
                'Metric': ['NPV', 'Cumulative Revenue', 'Average Monthly Revenue'],
                'Value': [
                    df['NPV'].iloc[-1] if 'NPV' in df else 0,
                    df['Net_Revenue'].sum(),
                    df['Net_Revenue'].mean()
                ]
            })
            financial_metrics.to_excel(writer, sheet_name='Financial Metrics', index=False)

        output_files['excel'] = excel_path

        return output_files


class Formatter:
    """Format values for display."""

    @staticmethod
    def format_currency(
            value: float,
            decimals: int = 2,
            prefix: str = "$",
            include_cents: bool = True
    ) -> str:
        """Format currency values."""
        if not include_cents:
            return f"{prefix}{int(value):,}"
        return f"{prefix}{value:,.{decimals}f}"

    @staticmethod
    def format_percentage(
            value: float,
            decimals: int = 1
    ) -> str:
        """Format percentage values."""
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_date(
            date: datetime,
            format_string: str = "%Y-%m-%d"
    ) -> str:
        """Format date values."""
        return date.strftime(format_string)


class Calculator:
    """Perform common calculations."""

    @staticmethod
    def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for a dataset."""
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'p10': np.percentile(data, 10),
            'p90': np.percentile(data, 90)
        }

    @staticmethod
    def calculate_ratios(
            values: np.ndarray,
            base: np.ndarray
    ) -> np.ndarray:
        """Calculate ratios between two arrays."""
        return np.divide(values, base, out=np.zeros_like(values), where=base != 0)

    @staticmethod
    def calculate_growth_rates(
            values: np.ndarray
    ) -> np.ndarray:
        """Calculate period-over-period growth rates."""
        return np.diff(values) / values[:-1]


class Logger:
    """Handle logging and error reporting."""

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.logs = []

    def log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} [{level}] {message}"
        self.logs.append(log_entry)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')

    def export_logs(self, output_path: Path):
        """Export logs to file."""
        with open(output_path, 'w') as f:
            for log in self.logs:
                f.write(log + '\n')


class CacheManager:
    """Manage data caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'well_analysis'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_data(self, data: Any, key: str):
        """Cache data with given key."""
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data for given key."""
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()

    def update(self, steps: int = 1):
        """Update progress."""
        self.current_step += steps

    def get_progress(self) -> Dict[str, Union[float, str]]:
        """Get progress information."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = self.current_step / self.total_steps

        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
        else:
            remaining = float('inf')

        return {
            'progress': progress * 100,
            'elapsed': f"{elapsed:.1f}s",
            'remaining': f"{remaining:.1f}s",
            'steps_completed': self.current_step,
            'total_steps': self.total_steps
        }