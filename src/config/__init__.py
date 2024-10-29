"""
Configuration management for oil well calculator.
"""

import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def load_all_configs():
    """Load all configuration files."""
    return {
        'equipment': load_config('equipment.yaml'),
        'regulations': load_config('regulations.yaml'),
        'tax_rates': load_config('tax_rates.yaml')
    }

def load_config(filename: str) -> dict:
    """Load a specific configuration file."""
    config_path = CONFIG_DIR / filename
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error loading configuration file {filename}: {str(e)}")

__all__ = ['load_all_configs', 'load_config']