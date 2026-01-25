"""
load_config.py
-------------------------
Merge base and trial-specific YAML configuration files for experiments.
"""

import yaml
from copy import deepcopy

def load_yaml(path):
    with open(path, "r") as f:
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
    base = load_yaml("configs/base.yaml")
    trial = load_yaml(trial_path)
    return merge_dicts(base, trial)
