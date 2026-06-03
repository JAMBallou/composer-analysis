"""
Generate labels CSV from extracted features
"""
import pandas as pd
from pathlib import Path
import sys

# Make `src` importable so we can use `utils.config`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.config import METADATA_CSV, AUDIO_FEATURES_DIR, LABELS_CSV

METADATA_PATH = METADATA_CSV
OUTPUT_DIR = AUDIO_FEATURES_DIR
LABELS_PATH = LABELS_CSV

# Load metadata
metadata = pd.read_csv(METADATA_PATH)
print(f"Loaded {len(metadata)} files from metadata")

# Check which files have corresponding features
label_rows = []
file_index = 0

for idx, row in metadata.iterrows():
    # Check if feature files exist for this file
    idx_str = f"{file_index:04d}"
    feat_file = OUTPUT_DIR / f"{idx_str}_mel_start.npy"
    
    if feat_file.exists():
        label_rows.append({
            "id": idx_str,
            "audio_filename": row["audio_filename"],
            "composer": row.get("canonical_composer", "unknown"),
            "period": row.get("period", "unknown")
        })
        file_index += 1

# Save labels
print(f"Saving {len(label_rows)} labels to {LABELS_PATH}")
pd.DataFrame(label_rows).to_csv(LABELS_PATH, index=False)
print("Done!")
print(f"\nExtraction Summary:")
print(f"Total metadata files: {len(metadata)}")
print(f"Successfully extracted: {len(label_rows)}")
print(f"Skipped: {len(metadata) - len(label_rows)}")
