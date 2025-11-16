"""
load_data.py
-------------
Load MAESTRO dataset metadata (CSV and JSON) into usable Python structures (df and dict).
"""

import pandas as pd
import json

def load_maestro_metadata(json_path: str):
    """
    Loads the MAESTRO dataset JSON metadata file.
    
    Args:
        json_path (str): Path to the MAESTRO JSON metadata file. 
            Currently: ``C:\Users\jamba\OneDrive\Documents\programming\composer-classification\data\maestro\maestro-v3.0.0.json``
    
    Returns:
        metadata (dict): Parsed JSON metadata as a Python dictionary.
    """

    # TODO: implement caching if loading repeatedly
    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        metadata = json.load(f)
    return metadata


def load_maestro_csv(csv_path: str):
    """
    Loads the MAESTRO dataset CSV metadata file into a pandas DataFrame.
    
    Args:
        csv_path (str): Path to the MAESTRO CSV metadata file.
            Currently: ``C:\Users\jamba\OneDrive\Documents\programming\composer-classification\data\maestro\maestro-v3.0.0.csv``

    Returns:
        df (pd.DataFrame): ``df`` containing the MAESTRO metadata.
    """

    df = pd.read_csv(csv_path, encoding="utf-8", error_bad_lines=False)
    # TODO: clean up columns, drop unused data
    return df

# Example usage
if __name__ == "__main__":
    print(load_maestro_metadata("data\maestro\maestro-v3.0.0.json"))   