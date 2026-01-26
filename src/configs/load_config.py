"""
load_config.py
-------------------------
Merge base and trial-specific YAML configuration files for experiments.
"""

import yaml
from copy import deepcopy
from pathlib import Path

def load_yaml(path):
    """Load YAML file from given path with UTF-8 encoding."""
    with open(path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def merge_dicts(base, override):
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result:
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result

def load_experiment_config(trial_path):
    """Load base config and merge with trial-specific config."""
    # Get the directory where this file is located
    config_dir = Path(__file__).resolve().parent
    base_path = config_dir / "base.yaml"
    
    # Handle trial_path - convert to absolute if relative
    trial_path = Path(trial_path)
    if not trial_path.is_absolute():
        # First check if it's relative to the config directory
        candidate = config_dir / trial_path.name
        if candidate.exists():
            trial_path = candidate
        else:
            # Otherwise resolve from project root
            project_root = config_dir.parent.parent
            trial_path = project_root / trial_path
    
    base = load_yaml(str(base_path))
    trial = load_yaml(str(trial_path))
    return merge_dicts(base, trial)
