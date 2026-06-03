"""
calculate_average_duration.py
-------------------------
Calculate the average duration of audio samples in the MAESTRO dataset.
"""

import pandas as pd
from pathlib import Path
import sys

# Make `src` importable so we can use `utils.config`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.config import METADATA_CSV

METADATA_PATH = METADATA_CSV

# Load metadata
df = pd.read_csv(METADATA_PATH)

# Calculate statistics
total_duration = df['duration'].sum()
average_duration = df['duration'].mean()
median_duration = df['duration'].median()
min_duration = df['duration'].min()
max_duration = df['duration'].max()
std_duration = df['duration'].std()
num_samples = len(df)

# Convert to minutes for readability
total_hours = total_duration / 3600
average_minutes = average_duration / 60
median_minutes = median_duration / 60
min_minutes = min_duration / 60
max_minutes = max_duration / 60

print("=" * 60)
print("MAESTRO Dataset Audio Duration Statistics")
print("=" * 60)
print(f"Total samples: {num_samples}")
print(f"\nTotal duration: {total_hours:.2f} hours ({total_duration:.2f} seconds)")
print(f"\nAverage duration: {average_minutes:.2f} minutes ({average_duration:.2f} seconds)")
print(f"Median duration: {median_minutes:.2f} minutes ({median_duration:.2f} seconds)")
print(f"Std deviation: {std_duration / 60:.2f} minutes ({std_duration:.2f} seconds)")
print(f"\nMin duration: {min_minutes:.2f} minutes ({min_duration:.2f} seconds)")
print(f"Max duration: {max_minutes:.2f} minutes ({max_duration:.2f} seconds)")
print("=" * 60)
