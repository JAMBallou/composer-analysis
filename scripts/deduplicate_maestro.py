"""
deduplicate_maestro.py
----------------------
Clean up duplicate works in the MAESTRO dataset.

The MAESTRO dataset contains multiple performances of the same works by different
pianists, years, etc. This script identifies and handles duplicates.

Usage:
    python scripts/deduplicate_maestro.py [--strategy STRATEGY] [--output OUTPUT]

Strategies:
    - first: Keep only the first occurrence (default, reproducible)
    - random: Keep one random occurrence from each duplicate group
    - best_quality: Keep the one with median duration (assumes more complete)
    - keep_all: Show duplicates but don't remove
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_maestro_data(csv_path: str) -> pd.DataFrame:
    """Load the MAESTRO CSV file."""
    return pd.read_csv(csv_path)


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
    # Convert to lowercase
    title = title.lower()
    
    # Normalize hyphenation: various hyphen types to single hyphen
    title = title.replace('–', '-')  # en-dash to hyphen
    title = title.replace('—', '-')  # em-dash to hyphen
    
    # Normalize spaces around punctuation
    title = title.replace(' , ', ', ')
    title = title.replace(' , ', ', ')
    
    # Normalize "WTC" variations (Well-Tempered Clavier)
    title = title.replace('wtc', 'wtc')  # Already lowercase
    title = title.replace('book i', 'wtc i')
    title = title.replace('book ii', 'wtc ii')
    title = title.replace('book iii', 'wtc iii')
    title = title.replace('book iv', 'wtc iv')
    
    # Normalize musical note notations
    # Convert "C sharp" and variations to "c#"
    title = title.replace('c sharp', 'c#')
    title = title.replace('d sharp', 'd#')
    title = title.replace('e sharp', 'e#')
    title = title.replace('f sharp', 'f#')
    title = title.replace('g sharp', 'g#')
    title = title.replace('a sharp', 'a#')
    title = title.replace('b sharp', 'b#')
    
    title = title.replace('c-sharp', 'c#')
    title = title.replace('d-sharp', 'd#')
    title = title.replace('e-sharp', 'e#')
    title = title.replace('f-sharp', 'f#')
    title = title.replace('g-sharp', 'g#')
    title = title.replace('a-sharp', 'a#')
    title = title.replace('b-sharp', 'b#')
    
    # Flats
    title = title.replace('c flat', 'cb')
    title = title.replace('d flat', 'db')
    title = title.replace('e flat', 'eb')
    title = title.replace('f flat', 'fb')
    title = title.replace('g flat', 'gb')
    title = title.replace('a flat', 'ab')
    title = title.replace('b flat', 'bb')
    
    title = title.replace('c-flat', 'cb')
    title = title.replace('d-flat', 'db')
    title = title.replace('e-flat', 'eb')
    title = title.replace('f-flat', 'fb')
    title = title.replace('g-flat', 'gb')
    title = title.replace('a-flat', 'ab')
    title = title.replace('b-flat', 'bb')
    
    # Normalize major/minor
    title = title.replace('major', 'maj')
    title = title.replace('minor', 'min')
    
    # Remove versioning info that doesn't matter (Complete, Version, etc.)
    title = title.replace('(complete)', '')
    title = title.replace(' - complete', '')
    title = title.replace(' complete', '')
    
    # Remove BWV numbers since they're catalog numbers (optional, for matching)
    # Just normalize spacing after "bwv"
    import re
    title = re.sub(r'bwv\s+', 'bwv ', title)
    
    # Remove extra spaces and normalize whitespace
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
            print(f"       Audio: {row['audio_filename']}")
    
    if len(duplicates) > limit:
        print(f"\n... and {len(duplicates) - limit} more duplicate works")


def deduplicate_first(df: pd.DataFrame, duplicates: Dict) -> pd.DataFrame:
    """
    Strategy: Keep only the first occurrence of each duplicate.
    
    This is reproducible and deterministic but may not be optimal.
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
    
    This heuristic assumes that more complete recordings tend to have middle-range
    durations, while incomplete or partial recordings are outliers.
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


def main():
    import sys
    # Make `src` importable so we can use `utils.config`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.config import METADATA_CSV

    parser = argparse.ArgumentParser(
        description="Clean up duplicate works in MAESTRO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show duplicates and use 'first' strategy
  python scripts/deduplicate_maestro.py --strategy first --output data/maestro/maestro-v3.0.0.csv
  
  # Use 'best_quality' strategy (keep recordings closest to median duration)
  python scripts/deduplicate_maestro.py --strategy best_quality --output data/maestro/maestro-v3.0.0.csv
  
  # Just show duplicates without saving
  python scripts/deduplicate_maestro.py --strategy keep_all
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=str(METADATA_CSV),
        help='Input MAESTRO CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (if not specified, no file is saved)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['first', 'best_quality', 'random', 'keep_all', 'keep_train'],
        default='first',
        help='Deduplication strategy to use'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (only for random strategy)'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all duplicates (not just first 10)'
    )
    parser.add_argument(
        '--also-backup',
        action='store_true',
        help='Also save the original file as backup'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading MAESTRO data from {args.input}...")
    df = load_maestro_data(args.input)
    print(f"✓ Loaded {len(df)} rows")
    
    # Find duplicates
    duplicates = find_duplicates(df)
    
    # Print statistics
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
    
    # Print duplicate details
    if args.show_all:
        print_duplicate_details(df, duplicates, limit=len(duplicates))
    else:
        print_duplicate_details(df, duplicates, limit=10)
    
    # Apply deduplication strategy
    print("\n" + "="*80)
    print(f"APPLYING DEDUPLICATION STRATEGY: {args.strategy.upper()}")
    print("="*80)
    
    if args.strategy == 'first':
        df_dedup = deduplicate_first(df, duplicates)
    elif args.strategy == 'best_quality':
        df_dedup = deduplicate_best_quality(df, duplicates)
    elif args.strategy == 'random':
        df_dedup = deduplicate_random(df, duplicates, args.random_seed)
    elif args.strategy == 'keep_train':
        df_dedup = deduplicate_keep_train(df, duplicates)
    else:  # keep_all
        print("\n✓ Keeping all duplicates (no changes made)")
        df_dedup = df.copy()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original rows: {len(df)}")
    print(f"Deduplicated rows: {len(df_dedup)}")
    print(f"Rows removed: {len(df) - len(df_dedup)}")
    print(f"Reduction: {100 * (len(df) - len(df_dedup)) / len(df):.1f}%")
    
    # Verify no new duplicates
    duplicates_after = find_duplicates(df_dedup)
    if len(duplicates_after) == 0:
        print("\n✓ Verification passed: No remaining duplicates!")
    else:
        print(f"\n⚠ Warning: {len(duplicates_after)} duplicate works remain")
    
    # Save output
    if args.output:
        # Backup original if requested
        if args.also_backup:
            backup_path = args.input.replace('.csv', '.bak.csv')
            print(f"\nBacking up original to: {backup_path}")
            df.to_csv(backup_path, index=False)
        
        print(f"\nSaving deduplicated dataset to: {args.output}")
        df_dedup.to_csv(args.output, index=False)
        print("✓ Successfully saved!")
        
        # Print summary of what was kept
        print(f"\nDeduplication complete!")
        print(f"Original: {args.input}")
        print(f"Output: {args.output}")
    else:
        print("\n(No output file specified, changes not saved)")
    
    # Save deduplication report
    report = {
        'original_rows': len(df),
        'deduplicated_rows': len(df_dedup),
        'rows_removed': len(df) - len(df_dedup),
        'reduction_percent': 100 * (len(df) - len(df_dedup)) / len(df),
        'strategy_used': args.strategy,
        'duplicate_works_found': stats['total_duplicates'],
        'top_composers': stats['top_composers_with_duplicates']
    }
    
    if args.output:
        report_path = args.output.replace('.csv', '_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
