"""
sync_csv.py
----------------------
One-time helper: backup and update MAESTRO CSV to remove rows that reference
audio/midi files that are missing on disk. Intended to be safe: creates a
timestamped backup before writing.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
# Make `src` importable so we can use `utils.config`
sys.path.insert(0, str(ROOT / "src"))
from utils.config import METADATA_CSV, MAESTRO_DIR

CSV = METADATA_CSV
DATA_DIR = MAESTRO_DIR / "data"

if not CSV.exists():
    print(f"CSV not found: {CSV}")
    sys.exit(2)
if not DATA_DIR.exists():
    print(f"Data directory not found: {DATA_DIR}")
    sys.exit(2)

print(f"Loading CSV: {CSV}")
df = pd.read_csv(CSV)
orig_len = len(df)

# normalize helper

def norm_path(s):
    if pd.isna(s):
        return ""
    p = str(s).strip()
    if p == "":
        return ""
    return Path(p).as_posix().lstrip("./").lstrip("/")

missing_rows = []
keep_mask = []
for _, row in df.iterrows():
    audio = norm_path(row.get("audio_filename", ""))
    midi = norm_path(row.get("midi_filename", ""))
    audio_ok = True
    midi_ok = True
    if audio:
        audio_path = DATA_DIR / audio
        audio_ok = audio_path.exists()
    if midi:
        midi_path = DATA_DIR / midi
        midi_ok = midi_path.exists()
    keep = audio_ok and midi_ok
    keep_mask.append(keep)
    if not keep:
        missing_rows.append({
            "audio": audio,
            "midi": midi,
        })

keep_mask = pd.Series(keep_mask)
removed_count = int((~keep_mask).sum())
print(f"Found {removed_count} rows referencing missing files out of {orig_len} rows.")

if removed_count == 0:
    print("Nothing to do. Exiting.")
    sys.exit(0)

# backup CSV
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
backup = CSV.with_suffix(f".bak.{ts}")
print(f"Backing up CSV to {backup}")
backup.write_bytes(CSV.read_bytes())

# write filtered CSV
new_df = df[keep_mask]
new_df.to_csv(CSV, index=False)
print(f"Wrote updated CSV: {CSV} (removed {removed_count} rows)")

# write audit of removed rows
audit = CSV.with_name(f"removed_rows.{ts}.csv")
import csv
with open(audit, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["audio", "midi"])
    writer.writeheader()
    for r in missing_rows:
        writer.writerow(r)
print(f"Wrote audit of removed rows to {audit}")
