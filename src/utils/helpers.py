import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_inputs(inputs: Dict[str, Any]) -> bool:
    """
    Validate input parameters.
    
    Args:
        inputs: Dictionary of input parameters
    
    Returns:
        True if inputs are valid
        
    Raises:
        ValueError: If inputs are invalid
    """
    required_fields = [
        'initial_production',
        'decline_rate',
        'b_factor'
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in inputs:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ranges
    if inputs['initial_production'] <= 0:
        raise ValueError("Initial production must be positive")
    
    if not 0 <= inputs['decline_rate'] <= 1:
        raise ValueError("Decline rate must be between 0 and 1")
    
    if not 0 <= inputs['b_factor'] <= 1:
        raise ValueError("B-factor must be between 0 and 1")
    
    return True

def format_currency(value: float,
                   decimals: int = 2,
                   prefix: str = "$",
                   include_cents: bool = True) -> str:
    """
    Format currency values.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        prefix: Currency prefix
        include_cents: Whether to include cents
    
    Returns:
        Formatted currency string
    """
    if not include_cents:
        value = round(value)
        return f"{prefix}{value:,.0f}"
    return f"{prefix}{value:,.{decimals}f}"

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical measures for a dataset.
    
    Args:
        data: Numeric array
    
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'p10': np.percentile(data, 10),
        'p90': np.percentile(data, 90)
    }

def export_results(
    df: pd.DataFrame,
    results_dir: Union[str, Path],
    prefix: Optional[str] = None
) -> Dict[str, Path]:
    """
    Export analysis results to files.
    
    Args:
        df: DataFrame with results
        results_dir: Directory for output files
        prefix: Optional prefix for filenames
    
    Returns:
        Dictionary of output file paths
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{prefix}_" if prefix else ""
    
    # Export to different formats
    output_files = {}
    
    # CSV export
    csv_path = results_dir / f"{prefix}results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    output_files['csv'] = csv_path
    
    # Excel export with multiple sheets
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
                df['NPV'].iloc[-1],
                df['Net_Revenue'].sum(),
                df['Net_Revenue'].mean()
            ]
        })
        financial_metrics.to_excel(writer, sheet_name='Financial Metrics', index=False)
    
    output_files['excel'] = excel_path
    
    return output_files