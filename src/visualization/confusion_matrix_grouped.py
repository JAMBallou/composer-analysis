"""
confusion_matrix_grouped.py
-------------------------
Create confusion matrices grouped by composer region and period instead of individual composers.

Usage:
python -m src.visualization.confusion_matrix_grouped <path_to_confusion_matrix>.npy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_composer_info():
    """Load composer metadata (period, region) from CSV."""
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "maestro" / "composer_info.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Composer info CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Create mapping: composer_name -> {period, region}
    composer_map = {}
    for _, row in df.iterrows():
        composer_map[row["Name"]] = {
            "period": row["Period"],
            "region": row["Region"]
        }
    
    return composer_map


def map_labels_to_attribute(labels, composer_map, attribute):
    """
    Map composer labels to period or region.
    
    Args:
        labels: list of composer names
        composer_map: dict mapping composer names to metadata
        attribute: 'period' or 'region'
    
    Returns:
        list of attribute values for each composer
    """
    result = []
    for label in labels:
        if label in composer_map:
            result.append(composer_map[label][attribute])
        else:
            result.append(f"Unknown {attribute}")
    
    return result


def aggregate_confusion_matrix(cm, original_labels, new_labels):
    """
    Aggregate confusion matrix by grouping composers into categories.
    
    Args:
        cm: original confusion matrix (n_composers x n_composers)
        original_labels: original composer names
        new_labels: new category labels (period or region) for each composer
    
    Returns:
        aggregated_cm: confusion matrix grouped by categories
        unique_labels: sorted unique category names
    """
    # Get unique categories in sorted order
    unique_labels = sorted(set(new_labels))
    n_categories = len(unique_labels)
    
    # Create label to index mapping
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # Create aggregated confusion matrix
    aggregated_cm = np.zeros((n_categories, n_categories), dtype=int)
    
    # Aggregate the original confusion matrix
    for i, true_label in enumerate(original_labels):
        for j, pred_label in enumerate(original_labels):
            true_category = new_labels[i]
            pred_category = new_labels[j]
            
            true_idx = label_to_idx[true_category]
            pred_idx = label_to_idx[pred_category]
            
            aggregated_cm[true_idx, pred_idx] += cm[i, j]
    
    return aggregated_cm, unique_labels


def plot_confusion_matrix(cm, labels, title, save_path, normalize=False):
    """Plot and save a confusion matrix."""
    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype(float) / row_sums
        cm_plot = cm_normalized
    else:
        cm_plot = cm

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot, cmap='Blues', aspect='auto')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm_plot.max() / 2

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            plt.text(
                j,
                i,
                format(cm_plot[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=10
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def process_confusion_matrix(npy_file):
    """Process a confusion matrix and create grouped versions by period and region."""
    
    if not os.path.exists(npy_file):
        print(f"Error: File not found: {npy_file}")
        sys.exit(1)
    
    if not npy_file.endswith('.npy'):
        print("Error: File must be a .npy file")
        sys.exit(1)
    
    # Load confusion matrix
    cm = np.load(npy_file)
    
    # Try to load composer labels from corresponding .json file
    json_file = npy_file.replace('.npy', '.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        composer_labels = data.get("labels", None)
    else:
        composer_labels = None
    
    if composer_labels is None:
        print("Error: Could not find composer labels in corresponding JSON file")
        print(f"Expected: {json_file}")
        sys.exit(1)
    
    # Load composer metadata
    try:
        composer_map = load_composer_info()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Output directory is the same as the input file
    output_dir = os.path.dirname(npy_file)
    if not output_dir:
        output_dir = "."
    
    # ===== Process by Period =====
    period_labels = map_labels_to_attribute(composer_labels, composer_map, 'period')
    period_cm, unique_periods = aggregate_confusion_matrix(cm, composer_labels, period_labels)
    
    # Plot normalized period confusion matrix
    plot_confusion_matrix(
        period_cm,
        unique_periods,
        title="Normalized Confusion Matrix by Period",
        save_path=os.path.join(output_dir, "confusion_matrix_by_period_normalized.png"),
        normalize=True
    )
    
    # ===== Process by Region =====
    region_labels = map_labels_to_attribute(composer_labels, composer_map, 'region')
    region_cm, unique_regions = aggregate_confusion_matrix(cm, composer_labels, region_labels)
    
    # Plot normalized region confusion matrix
    plot_confusion_matrix(
        region_cm,
        unique_regions,
        title="Normalized Confusion Matrix by Region",
        save_path=os.path.join(output_dir, "confusion_matrix_by_region_normalized.png"),
        normalize=True
    )
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Period categories: {unique_periods}")
    print(f"Region categories: {unique_regions}")
    print(f"\nAll files saved to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.visualization.confusion_matrix_grouped <path_to_confusion_matrix.npy>")
        sys.exit(1)
    
    npy_file = sys.argv[1]
    process_confusion_matrix(npy_file)
