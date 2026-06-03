import pandas as pd
from pathlib import Path
import sys

# Make `src` importable so we can use `utils.config`
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.config import METADATA_CSV, MAESTRO_DIR

# Check CSV
df = pd.read_csv(METADATA_CSV)
print("CSV audio_filename samples:")
for i in range(3):
    print(f"  {df['audio_filename'].iloc[i]}")

# Check what script constructs
DATASET_DIR = MAESTRO_DIR / "data"
print(f"\nDATASET_DIR: {DATASET_DIR}")

# Build path as script does
test_path = DATASET_DIR / df['audio_filename'].iloc[0]
print(f"\nPath script builds:")
print(f"  {test_path}")
print(f"  Exists: {test_path.exists()}")

# Check what actually exists
actual_path = REPO_ROOT / "data" / "maestro" / "data" / df['audio_filename'].iloc[0]
print(f"\nActual path (should be same):")
print(f"  {actual_path}")
print(f"  Exists: {actual_path.exists()}")

# Check first real file
import os
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith('.wav'):
            print(f"\nFirst actual WAV file found:")
            full = Path(root) / file
            print(f"  {full}")
            rel = full.relative_to(DATASET_DIR)
            print(f"  Relative to DATASET_DIR: {rel}")
            break
    if files:
        break
