"""
count_composers.py
==================
Counts the number of works from each composer in the MAESTRO dataset.

Creates two files:
1. composer_count.csv - counts from the original/backup dataset
2. composer_count_filtered.csv - counts from the current filtered dataset
"""

import pandas as pd
from pathlib import Path


def count_composers(csv_path: Path, output_path: Path):
    """
    Count works per composer and save to CSV.
    
    Args:
        csv_path: Path to the MAESTRO CSV file
        output_path: Path where to save the composer count CSV
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Count works per composer
    composer_counts = df["canonical_composer"].value_counts()
    
    # Convert to DataFrame with proper column names
    count_df = pd.DataFrame({
        "Composer": composer_counts.index,
        "Count": composer_counts.values
    })
    
    # Save to CSV
    count_df.to_csv(output_path, index=False)
    print(f"Saved {len(count_df)} composers to {output_path}")
    
    # Print summary
    print(f"\nTop 10 composers:")
    for idx, row in count_df.head(10).iterrows():
        print(f"  {row['Composer']}: {row['Count']}")
    
    return count_df


def main():
    # Get paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    maestro_dir = repo_root / "data" / "maestro"
    
    # Original dataset (backup)
    backup_csv = maestro_dir / "maestro-v3.0.0.bak.csv"
    output_original = maestro_dir / "composer_count.csv"
    
    # Current filtered dataset
    current_csv = maestro_dir / "maestro-v3.0.0.csv"
    output_filtered = maestro_dir / "composer_count_filtered.csv"
    
    print("="*80)
    print("COUNTING COMPOSERS IN MAESTRO DATASET")
    print("="*80)
    
    # Count from original dataset
    if backup_csv.exists():
        print(f"\n1. Processing original dataset: {backup_csv}")
        count_composers(backup_csv, output_original)
    else:
        print(f"\n1. Backup file not found: {backup_csv}")
        print("   Skipping composer_count.csv generation")
    
    # Count from filtered dataset
    if current_csv.exists():
        print(f"\n2. Processing current filtered dataset: {current_csv}")
        count_composers(current_csv, output_filtered)
    else:
        print(f"\n2. Current CSV not found: {current_csv}")
        print("   Skipping composer_count_filtered.csv generation")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
