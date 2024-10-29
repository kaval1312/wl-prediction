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

def load_config(filename):
    """Load a specific configuration file."""
    config_path = CONFIG_DIR / filename
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
