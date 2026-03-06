"""
Generate labels CSV from extracted features - using only standard library
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = REPO_ROOT / "data" / "maestro" / "maestro-v3.0.0.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "features" / "audio"
LABELS_PATH = REPO_ROOT / "outputs" / "features" / "labels.csv"

# Read metadata CSV
metadata_rows = []
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    metadata_rows = list(reader)

print(f"Loaded {len(metadata_rows)} files from metadata")

# Check which files have corresponding features
label_rows = []
file_index = 0

for row in metadata_rows:
    # Check if feature files exist for this file
    idx_str = f"{file_index:04d}"
    feat_file = OUTPUT_DIR / f"{idx_str}_mel_start.npy"
    
    if feat_file.exists():
        label_rows.append({
            "id": idx_str,
            "audio_filename": row.get("audio_filename", ""),
            "composer": row.get("canonical_composer", "unknown"),
            "period": row.get("period", "unknown"),
            "midi_filename": row.get("midi_filename", "")
        })
        file_index += 1

# Write labels CSV
with open(LABELS_PATH, 'w', newline='', encoding='utf-8') as f:
    if label_rows:
        fieldnames = label_rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(label_rows)

print(f"\n{'='*60}")
print("LABELS CSV CREATED")
print(f"{'='*60}")
print(f"Total extracted features: {len(label_rows)}")
print(f"Saved to: {LABELS_PATH}")
print(f"{'='*60}")

# Show first few rows
if label_rows:
    print("\nFirst 3 rows:")
    for row in label_rows[:3]:
        print(f"  {row['id']}: {row['composer']} - {row['audio_filename'][:50]}")
