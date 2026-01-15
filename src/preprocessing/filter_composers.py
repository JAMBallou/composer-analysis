"""
filter_composers.py
-------------------
Filters MAESTRO dataset to specific composers or periods. Also includes a one-time utility to remove unneeded audio/midi files from dataset.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Set
import shutil
from datetime import datetime

def filter_top_composers(df: pd.DataFrame, top_n: int = 14):
    """
    Filters ``df`` to include only the ``top_n`` most frequent composers.

    Args:
        df (pd.DataFrame): ``df`` containing MAESTRO metadata.
        top_n (int): Number of composers to keep.

    Returns:
        filtered (pd.DataFrame): DataFrame with only the ``top_n`` composers.
    """

    top_composers = df["canonical_composer"].value_counts().head(top_n).index
    filtered = df[df["canonical_composer"].isin(top_composers)]
    return filtered

# one-time executable script to remove unnecessary files from the dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "One-time utility: remove WAV/MIDI files from the MAESTRO dataset that are NOT "
            "referenced by the filtered metadata DataFrame. By default this runs as a dry-run."
        )
    )

    # Optional arguments
    parser.add_argument("--csv", default=str(Path(__file__).resolve().parents[2] / "data" / "maestro" / "maestro-v3.0.0.csv"), help="Path to maestro CSV metadata")
    parser.add_argument("--maestro-root", default=str(Path(__file__).resolve().parents[2] / "data" / "maestro"), help="Path to maestro root directory (contains 'data/' with audio/midi files)")
    parser.add_argument("--top-n", type=int, default=14, help="Keep top-N composers when filtering")
    parser.add_argument("--delete", action="store_true", help="Actually delete files. Without this flag the script does a dry-run and only reports what would be removed.")
    parser.add_argument("--yes", action="store_true", help="If set with --delete, skip confirmation prompt and proceed")
    parser.add_argument("--update-csv", action="store_true", help="If set with --delete, remove references to deleted files from the CSV (after backup if requested)")
    parser.add_argument("--backup-csv", action="store_true", help="When updating CSV, create a timestamped backup before writing")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    maestro_root = Path(args.maestro_root)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(2)

    # load CSV metadata as df to find unneeded files
    df = pd.read_csv(csv_path)
    filtered_df = filter_top_composers(df, top_n=args.top_n)

    # build allowed sets from CSV (these are paths relative to maestro_root / 'data')
    def _normalize(p: str) -> str:
        # Normalize to posix-style relative path without any leading ./ or / 
        return str(Path(p).as_posix()).lstrip("./").lstrip("/")

    allowed_audio: Set[str] = set(filtered_df["audio_filename"].dropna().astype(str).map(_normalize).tolist())
    allowed_midi: Set[str] = set(filtered_df["midi_filename"].dropna().astype(str).map(_normalize).tolist())

    data_tree = maestro_root / "data"
    if not data_tree.exists():
        print(f"ERROR: expected maestro data directory not found: {data_tree}")
        sys.exit(2)

    # Collect candidate files to consider for deletion (wav, midi, mid)
    to_delete = []
    total_candidates = 0
    for p in data_tree.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in {".wav", ".midi", ".mid"}:
            continue
        total_candidates += 1
        # relative path (posix) relative to data_tree
        try:
            rel = p.relative_to(data_tree).as_posix()
        except Exception:
            # fallback to filename only
            rel = p.name

        # decide whether this file is referenced in the filtered CSV
        if suffix == ".wav":
            if rel not in allowed_audio:
                to_delete.append(p)
        else:
            # midi/.midi/.mid
            if rel not in allowed_midi:
                to_delete.append(p)

    print(f"Found {total_candidates} audio/midi files under {data_tree}")
    print(f"Allowed (referenced) audio files: {len(allowed_audio)}; midi files: {len(allowed_midi)}")
    print(f"Files that would be removed: {len(to_delete)}")
    if len(to_delete) > 0:
        # show up to 20 example paths
        print("Examples:")
        for p in to_delete[:20]:
            print("  ", p.relative_to(maestro_root).as_posix())

    if not args.delete:
        print("Dry-run mode (no files deleted). Re-run with --delete to remove the listed files.")
        sys.exit(0)

    # Delete mode: confirm
    if not args.yes:
        ans = input(f"Are you sure you want to permanently delete {len(to_delete)} files? [y/N]: ")
        if ans.lower() not in {"y", "yes"}:
            print("Aborted by user.")
            sys.exit(0)

    # perform deletions
    removed = 0
    errors = 0
    removed_paths = []
    for p in to_delete:
        try:
            p.unlink()
            removed += 1
            # record relative path inside data/ for CSV removal
            try:
                rel = p.relative_to(data_tree).as_posix()
            except Exception:
                rel = p.name
            removed_paths.append(rel)
        except Exception as e:
            print(f"Failed to remove {p}: {e}")
            errors += 1
    print(f"Deletion complete. Removed={removed}; errors={errors}")

    # Optionally update CSV to remove references to deleted files
    if args.update_csv:
        if removed == 0:
            print("No files removed; skipping CSV update.")
        else:
            # Backup CSV if requested
            if args.backup_csv:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup_path = csv_path.with_suffix(f".bak.{ts}")
                try:
                    shutil.copy2(csv_path, backup_path)
                    print(f"CSV backed up to {backup_path}")
                except Exception as e:
                    print(f"Failed to create CSV backup: {e}")

            # Normalize CSV path columns and drop rows referencing removed files
            df_orig_len = len(df)

            def _norm_series(series):
                return series.fillna("").astype(str).map(lambda s: Path(s).as_posix().lstrip("./").lstrip("/"))

            audio_norm = _norm_series(df.get("audio_filename", pd.Series([""] * len(df))))
            midi_norm = _norm_series(df.get("midi_filename", pd.Series([""] * len(df))))

            # build boolean mask: True for rows to KEEP
            keep_mask = ~audio_norm.isin(removed_paths) & ~midi_norm.isin(removed_paths)
            df_filtered = df[keep_mask]
            removed_rows = df_orig_len - len(df_filtered)

            try:
                df_filtered.to_csv(csv_path, index=False)
                print(f"CSV updated: removed {removed_rows} rows referencing deleted files.")
            except Exception as e:
                print(f"Failed to write updated CSV: {e}")