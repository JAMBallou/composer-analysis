"""
load_data.py
-------------
Load MAESTRO dataset metadata (CSV/JSON) into usable Python structures.
"""

import pandas as pd
import json

def load_maestro_metadata(json_path: str):
    """
    Loads the MAESTRO dataset metadata file (JSON).
    """
    # TODO: implement caching if loading repeatedly
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata


def load_maestro_csv(csv_path: str):
    """
    Loads the MAESTRO CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    # TODO: clean up columns, drop unused data
    return df
