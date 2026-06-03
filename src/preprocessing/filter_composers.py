"""
filter_composers.py
-------------------
Filters MAESTRO dataset to specific composers or periods, with deduplication support.
Also includes a one-time utility to remove unneeded audio/midi files from dataset.

Deduplication strategies:
    - first: Keep only the first occurrence (default, reproducible)
    - random: Keep one random occurrence from each duplicate group
    - best_quality: Keep the one with median duration (assumes more complete)
    - keep_train: Prefer recordings from 'train' split
    - keep_all: Show duplicates but don't remove
"""

import pandas as pd
import numpy as np
import argparse
import sys
import json
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict
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


def normalize_title(title: str) -> str:
    """
    Normalize a title for case-insensitive and format-insensitive comparison.
    
    Handles:
    - Case differences
    - Hyphen variations (- vs no space, hyphen vs en-dash)
    - Musical notation (sharp/flat, natural)
    - Work numbering variations (WTC I vs Book I)
    - BWV/opus number variations
    - Extra whitespace
    """
    import re
    
    # Convert to lowercase
    title = title.lower()
    
    # Normalize hyphenation: various hyphen types to single hyphen
    title = title.replace('–', '-')  # en-dash to hyphen
    title = title.replace('—', '-')  # em-dash to hyphen
    
    # Normalize spaces around punctuation
    title = title.replace(' , ', ', ')
    
    # Normalize "WTC" variations (Well-Tempered Clavier)
    title = title.replace('book i', 'wtc i')
    title = title.replace('book ii', 'wtc ii')
    title = title.replace('book iii', 'wtc iii')
    title = title.replace('book iv', 'wtc iv')
    
    # Normalize musical note notations
    # Sharps
    for note in ['c', 'd', 'e', 'f', 'g', 'a', 'b']:
        title = title.replace(f'{note} sharp', f'{note}#')
        title = title.replace(f'{note}-sharp', f'{note}#')
    
    # Flats
    for note in ['c', 'd', 'e', 'f', 'g', 'a', 'b']:
        title = title.replace(f'{note} flat', f'{note}b')
        title = title.replace(f'{note}-flat', f'{note}b')
    
    # Normalize major/minor
    title = title.replace('major', 'maj')
    title = title.replace('minor', 'min')
    
    # Remove versioning info
    title = title.replace('(complete)', '')
    title = title.replace(' - complete', '')
    title = title.replace(' complete', '')
    
    # Normalize BWV spacing
    title = re.sub(r'bwv\s+', 'bwv ', title)
    
    # Remove extra spaces
    title = ' '.join(title.split())
    
    return title


def find_duplicates(df: pd.DataFrame) -> Dict[Tuple, List]:
    """
    Find duplicate works (same composer and title, case-insensitive).
    
    Returns:
        Dictionary mapping (normalized_composer, normalized_title) to list of row indices
    """
    duplicates = defaultdict(list)
    
    for idx, row in df.iterrows():
        # Normalize both composer and title for comparison
        composer = row['canonical_composer'].lower().strip()
        title = normalize_title(row['canonical_title'])
        key = (composer, title)
        duplicates[key].append(idx)
    
    # Filter to only keep actual duplicates (more than 1 occurrence)
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    
    return duplicates


def get_dedup_stats(df: pd.DataFrame, duplicates: Dict) -> Dict:
    """Get statistics about duplicates."""
    total_duplicates = len(duplicates)
    total_duplicate_rows = sum(len(rows) for rows in duplicates.values())
    rows_to_remove = total_duplicate_rows - total_duplicates
    
    # Group by composer
    duplicates_by_composer = defaultdict(int)
    for (composer, title), rows in duplicates.items():
        duplicates_by_composer[composer] += len(rows)
    
    # Sort by frequency
    top_composers = sorted(
        duplicates_by_composer.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return {
        'total_duplicates': total_duplicates,
        'total_duplicate_rows': total_duplicate_rows,
        'rows_to_remove': rows_to_remove,
        'top_composers_with_duplicates': top_composers,
        'total_rows': len(df)
    }


def print_duplicate_details(df: pd.DataFrame, duplicates: Dict, limit: int = 10):
    """Print details about duplicate works."""
    print("\n" + "="*80)
    print("DUPLICATE WORKS IN MAESTRO DATASET")
    print("="*80)
    
    for i, ((composer, title), row_indices) in enumerate(list(duplicates.items())[:limit]):
        print(f"\n{i+1}. {composer} - {title}")
        print(f"   Occurrences: {len(row_indices)}")
        
        for idx, row_idx in enumerate(row_indices, 1):
            row = df.iloc[row_idx]
            print(f"   [{idx}] Year: {row['year']}, Duration: {row['duration']:.1f}s, Split: {row['split']}")
            print(f"       MIDI: {row['midi_filename']}")
    
    if len(duplicates) > limit:
        print(f"\n... and {len(duplicates) - limit} more duplicate works")


def deduplicate_first(df: pd.DataFrame, duplicates: Dict) -> pd.DataFrame:
    """
    Strategy: Keep only the first occurrence of each duplicate.
    This is reproducible and deterministic.
    """
    rows_to_drop = []
    
    for (composer, title), row_indices in duplicates.items():
        # Keep the first one, drop the rest
        rows_to_drop.extend(row_indices[1:])
    
    df_dedup = df.drop(index=rows_to_drop).reset_index(drop=True)
    
    print(f"\n✓ Kept first occurrence of each duplicate work")
    print(f"  Removed {len(rows_to_drop)} duplicate rows")
    
    return df_dedup


def deduplicate_best_quality(df: pd.DataFrame, duplicates: Dict) -> pd.DataFrame:
    """
    Strategy: Keep the one with median duration (assumes most complete performance).
    """
    rows_to_drop = []
    
    for (composer, title), row_indices in duplicates.items():
        # Calculate durations
        durations = df.loc[row_indices, 'duration'].values
        
        # Find the one closest to median duration
        median_duration = np.median(durations)
        
        closest_idx = row_indices[
            np.argmin(np.abs(durations - median_duration))
        ]
        
        # Keep the closest to median, drop others
        for idx in row_indices:
            if idx != closest_idx:
                rows_to_drop.append(idx)
    
    df_dedup = df.drop(index=rows_to_drop).reset_index(drop=True)
    
    print(f"\n✓ Kept recordings closest to median duration")
    print(f"  Removed {len(rows_to_drop)} duplicate rows")
    
    return df_dedup


def deduplicate_random(df: pd.DataFrame, duplicates: Dict, random_seed: int = 42) -> pd.DataFrame:
    """
    Strategy: Keep one random occurrence from each duplicate group.
    """
    np.random.seed(random_seed)
    rows_to_drop = []
    
    for (composer, title), row_indices in duplicates.items():
        # Randomly select one to keep, drop others
        keep_idx = np.random.choice(row_indices)
        
        for idx in row_indices:
            if idx != keep_idx:
                rows_to_drop.append(idx)
    
    df_dedup = df.drop(index=rows_to_drop).reset_index(drop=True)
    
    print(f"\n✓ Kept random occurrence from each duplicate group")
    print(f"  Removed {len(rows_to_drop)} duplicate rows")
    print(f"  (Random seed: {random_seed})")
    
    return df_dedup


def deduplicate_keep_train(df: pd.DataFrame, duplicates: Dict) -> pd.DataFrame:
    """
    Strategy: Prefer recordings from 'train' split when duplicates exist.
    Otherwise keep the first one.
    """
    rows_to_drop = []
    
    for (composer, title), row_indices in duplicates.items():
        # Get splits for these rows
        rows_data = df.iloc[row_indices]
        
        # Check if any are in 'train' split
        train_indices = rows_data[rows_data['split'] == 'train'].index.tolist()
        
        if train_indices:
            # Keep the first training set occurrence
            keep_idx = train_indices[0]
        else:
            # Keep the first one overall
            keep_idx = row_indices[0]
        
        # Drop all others
        for idx in row_indices:
            if idx != keep_idx:
                rows_to_drop.append(idx)
    
    df_dedup = df.drop(index=rows_to_drop).reset_index(drop=True)
    
    print(f"\n✓ Preferred recordings from 'train' split")
    print(f"  Removed {len(rows_to_drop)} duplicate rows")
    
    return df_dedup


def apply_deduplication(df: pd.DataFrame, strategy: str = 'first', 
                       random_seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """
    Apply deduplication strategy to the dataframe.
    
    Args:
        df: Input dataframe
        strategy: Deduplication strategy ('first', 'best_quality', 'random', 'keep_train', 'keep_all')
        random_seed: Random seed for reproducibility (only for random strategy)
        verbose: Print detailed information
    
    Returns:
        Deduplicated dataframe
    """
    # Find duplicates
    duplicates = find_duplicates(df)
    
    if len(duplicates) == 0:
        if verbose:
            print("\n✓ No duplicates found in the dataset")
        return df
    
    # Print statistics if verbose
    if verbose:
        stats = get_dedup_stats(df, duplicates)
        
        print("\n" + "="*80)
        print("DEDUPLICATION STATISTICS")
        print("="*80)
        print(f"Total rows in dataset: {stats['total_rows']}")
        print(f"Duplicate works (composer+title combinations): {stats['total_duplicates']}")
        print(f"Total rows that are duplicates: {stats['total_duplicate_rows']}")
        print(f"Rows to remove: {stats['rows_to_remove']}")
        print(f"\nTop composers with duplicate works:")
        for composer, count in stats['top_composers_with_duplicates']:
            print(f"  - {composer}: {count} rows in duplicates")
        
        # Show some examples
        print_duplicate_details(df, duplicates, limit=5)
        
        print("\n" + "="*80)
        print(f"APPLYING DEDUPLICATION STRATEGY: {strategy.upper()}")
        print("="*80)
    
    # Apply strategy
    if strategy == 'first':
        df_dedup = deduplicate_first(df, duplicates)
    elif strategy == 'best_quality':
        df_dedup = deduplicate_best_quality(df, duplicates)
    elif strategy == 'random':
        df_dedup = deduplicate_random(df, duplicates, random_seed)
    elif strategy == 'keep_train':
        df_dedup = deduplicate_keep_train(df, duplicates)
    else:  # keep_all
        if verbose:
            print("\n✓ Keeping all duplicates (no changes made)")
        df_dedup = df.copy()
    
    # Verify no new duplicates
    if strategy != 'keep_all' and verbose:
        duplicates_after = find_duplicates(df_dedup)
        if len(duplicates_after) == 0:
            print("\n✓ Verification passed: No remaining duplicates!")
        else:
            print(f"\n⚠ Warning: {len(duplicates_after)} duplicate works remain")
    
    return df_dedup

# one-time executable script to remove unnecessary files from the dataset
if __name__ == "__main__":
    # Load config paths
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from utils.config import METADATA_CSV, MAESTRO_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Filter MAESTRO dataset by top composers and/or deduplicate works. "
            "Optionally remove WAV/MIDI files not referenced by the filtered metadata. "
            "By default runs as a dry-run when deleting files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter to top 14 composers and deduplicate with 'first' strategy
  python -m src.preprocessing.filter_composers --top-n 14 --deduplicate first
  
  # Filter and deduplicate, save to new file
  python -m src.preprocessing.filter_composers --top-n 14 --deduplicate best_quality --output data/maestro/maestro_filtered_dedup.csv
  
  # Just deduplicate without filtering
  python -m src.preprocessing.filter_composers --no-filter --deduplicate first --output data/maestro/maestro_dedup.csv
  
  # Filter and delete unreferenced files (with confirmation)
  python -m src.preprocessing.filter_composers --top-n 14 --delete
        """
    )

    # Input/output arguments
    parser.add_argument("--csv", default=str(METADATA_CSV), 
                       help="Path to maestro CSV metadata")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save filtered/deduplicated CSV (if not specified, updates original)")
    parser.add_argument("--maestro-root", default=str(MAESTRO_DIR), 
                       help="Path to maestro root directory (contains 'data/' with audio/midi files)")
    
    # Filtering arguments
    parser.add_argument("--top-n", type=int, default=12, 
                       help="Keep top-N composers when filtering")
    parser.add_argument("--no-filter", action="store_true",
                       help="Skip composer filtering (only deduplicate)")
    
    # Deduplication arguments
    parser.add_argument("--deduplicate", type=str, default=None,
                       choices=['first', 'best_quality', 'random', 'keep_train', 'keep_all'],
                       help="Deduplication strategy (if not specified, no deduplication)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility (only for random strategy)")
    parser.add_argument("--show-all-duplicates", action="store_true",
                       help="Show all duplicates (not just first 5)")
    
    # File deletion arguments
    parser.add_argument("--delete", action="store_true", 
                       help="Actually delete unreferenced files. Without this flag the script does a dry-run.")
    parser.add_argument("--yes", action="store_true", 
                       help="If set with --delete, skip confirmation prompt")
    parser.add_argument("--update-csv", action="store_true", 
                       help="If set with --delete, remove references to deleted files from the CSV")
    parser.add_argument("--backup-csv", action="store_true", 
                       help="Create timestamped backup before modifying CSV")
    
    args = parser.parse_args()

    csv_path = Path(args.csv)
    maestro_root = Path(args.maestro_root)
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(2)

    # Load CSV metadata
    print(f"Loading MAESTRO data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows")
    original_rows = len(df)
    
    # ===== Step 1: Filter by top composers =====
    if not args.no_filter:
        print(f"\n{'='*80}")
        print(f"FILTERING TO TOP {args.top_n} COMPOSERS")
        print(f"{'='*80}")
        filtered_df = filter_top_composers(df, top_n=args.top_n)
        print(f"✓ Filtered from {len(df)} to {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
        
        # Show composer distribution
        print("\nComposer distribution:")
        for composer, count in filtered_df["canonical_composer"].value_counts().items():
            print(f"  {composer}: {count} pieces")
        
        df = filtered_df
    else:
        print("\n✓ Skipping composer filtering (--no-filter specified)")
    
    # ===== Step 2: Deduplicate =====
    if args.deduplicate:
        df = apply_deduplication(
            df, 
            strategy=args.deduplicate, 
            random_seed=args.random_seed,
            verbose=True
        )
        
        print(f"\n{'='*80}")
        print("DEDUPLICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Original rows: {original_rows}")
        print(f"After filtering: {len(df)}")
        print(f"Total reduction: {100 * (original_rows - len(df)) / original_rows:.1f}%")
    else:
        print("\n✓ Skipping deduplication (no --deduplicate strategy specified)")
    
    # ===== Step 3: Save filtered/deduplicated CSV =====
    output_path = Path(args.output) if args.output else csv_path
    
    if args.output or args.deduplicate or not args.no_filter:
        # Create backup if requested
        if args.backup_csv and output_path.exists():
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = output_path.with_suffix(f".bak.{ts}.csv")
            try:
                shutil.copy2(output_path, backup_path)
                print(f"\n✓ CSV backed up to {backup_path}")
            except Exception as e:
                print(f"⚠ Failed to create CSV backup: {e}")
        
        # Save the filtered/deduplicated CSV
        print(f"\nSaving to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} rows to {output_path}")
        
        # Save processing report
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_rows': original_rows,
            'final_rows': len(df),
            'rows_removed': original_rows - len(df),
            'reduction_percent': 100 * (original_rows - len(df)) / original_rows,
            'filtering_applied': not args.no_filter,
            'top_n': args.top_n if not args.no_filter else None,
            'deduplication_strategy': args.deduplicate,
            'random_seed': args.random_seed if args.deduplicate == 'random' else None
        }
        
        report_path = output_path.with_suffix('.report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Processing report saved to {report_path}")
    
    # ===== Step 4: Delete unreferenced files (optional) =====
    # Note: This section analyzes files whether in dry-run or delete mode
    print(f"\n{'='*80}")
    print("FILE DELETION ANALYSIS")
    print(f"{'='*80}")
    
    # build allowed sets from the (possibly filtered/deduplicated) CSV
    def _normalize(p: str) -> str:
        # Normalize to posix-style relative path without any leading ./ or / 
        return str(Path(p).as_posix()).lstrip("./").lstrip("/")

    allowed_audio: Set[str] = set(df["audio_filename"].dropna().astype(str).map(_normalize).tolist())
    allowed_midi: Set[str] = set(df["midi_filename"].dropna().astype(str).map(_normalize).tolist())

    data_tree = maestro_root / "data"
    if not data_tree.exists():
        print(f"⚠ Maestro data directory not found: {data_tree}")
        print("  Skipping file deletion analysis")
    else:
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
            print("\nDry-run mode (no files deleted). Re-run with --delete to remove the listed files.")
        else:
            # Delete mode: confirm
            if len(to_delete) == 0:
                print("\n✓ No files to delete")
            else:
                if not args.yes:
                    ans = input(f"\nAre you sure you want to permanently delete {len(to_delete)} files? [y/N]: ")
                    if ans.lower() not in {"y", "yes"}:
                        print("Aborted by user.")
                        sys.exit(0)

                # perform deletions
                print(f"\nDeleting {len(to_delete)} files...")
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
                print(f"✓ Deletion complete. Removed={removed}; errors={errors}")

                # Optionally update CSV to remove references to deleted files
                if args.update_csv:
                    if removed == 0:
                        print("No files removed; skipping CSV update.")
                    else:
                        # Backup CSV if requested (if not already backed up)
                        if args.backup_csv and not (args.output or args.deduplicate or not args.no_filter):
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
                            df_filtered.to_csv(output_path, index=False)
                            print(f"CSV updated: removed {removed_rows} rows referencing deleted files.")
                        except Exception as e:
                            print(f"Failed to write updated CSV: {e}")
    
    # ===== Final Summary =====
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_path}")
    print(f"Original rows: {original_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Total reduction: {100 * (original_rows - len(df)) / original_rows:.1f}%")
    if args.deduplicate:
        print(f"Deduplication: {args.deduplicate}")
    if not args.no_filter:
        print(f"Filtered to top {args.top_n} composers")
    print(f"{'='*80}\n")
