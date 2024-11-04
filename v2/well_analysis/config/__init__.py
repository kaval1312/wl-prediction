from pathlib import Path
import yaml
import json

CONFIG_DIR = Path(__file__).parent

def load_hpc_config():
    with open(CONFIG_DIR / 'hpc_config.json') as f:
        return json.load(f)

def load_params_config():
    with open(CONFIG_DIR / 'params_config.yaml') as f:
        return yaml.safe_load(f)
