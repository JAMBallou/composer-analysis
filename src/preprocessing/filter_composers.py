"""
filter_composers.py
-------------------
Filters MAESTRO dataset to specific composers or periods.
"""

import pandas as pd

def filter_top_composers(df: pd.DataFrame, top_n: int = 14):
    """
    Filters DataFrame to include only top N most frequent composers.
    """
    top_composers = df["composer"].value_counts().head(top_n).index
    filtered = df[df["composer"].isin(top_composers)]
    return filtered
