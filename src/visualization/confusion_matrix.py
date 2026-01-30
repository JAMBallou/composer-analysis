"""
confusion_matrix.py
-------------------------
Visualization script for confusion matrices from composer classification models.

Shell command usage:
python -m src.visualization.confusion_matrix <path_to_confusion_matrix>.npy

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(
    cm,
    labels,
    title,
    save_path,
    normalize=False
):
    """Plot and save a confusion matrix."""
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrices_for_run(run_dir):
    """Plot and save confusion matrices for a specific trial run."""
    json_path = os.path.join(run_dir, "confusion_matrix.json")
    npy_path = os.path.join(run_dir, "confusion_matrix.npy")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        labels = data["labels"]
        cm = np.array(data["matrix"])
    elif os.path.exists(npy_path):
        cm = np.load(npy_path)
        labels = [f"Class {i}" for i in range(cm.shape[0])]
    else:
        return False

    # Raw confusion matrix
    plot_confusion_matrix(
        cm,
        labels,
        title="Confusion Matrix",
        save_path=os.path.join(run_dir, "confusion_matrix.png"),
        normalize=False
    )

    # Normalized confusion matrix
    plot_confusion_matrix(
        cm,
        labels,
        title="Normalized Confusion Matrix",
        save_path=os.path.join(run_dir, "confusion_matrix_normalized.png"),
        normalize=True
    )

    return True

def plot_all_confusion_matrices(run_dir):
    """Plot confusion matrices for a specific run directory."""
    if plot_confusion_matrices_for_run(run_dir):
        print(f"Saved confusion matrices to {run_dir}")
    else:
        print(f"No confusion matrix found in {run_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode: plot confusion matrix from specified .npy file
        npy_file = sys.argv[1]
        
        if not os.path.exists(npy_file):
            print(f"Error: File not found: {npy_file}")
            sys.exit(1)
        
        if not npy_file.endswith('.npy'):
            print("Error: File must be a .npy file")
            sys.exit(1)
        
        # Load confusion matrix
        cm = np.load(npy_file)
        
        # Try to load labels from corresponding .json file
        json_file = npy_file.replace('.npy', '.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            labels = data.get("labels", [f"Class {i}" for i in range(cm.shape[0])])
        else:
            labels = [f"Class {i}" for i in range(cm.shape[0])]
        
        # Output directory is the same as the input file
        output_dir = os.path.dirname(npy_file)
        if not output_dir:
            output_dir = "."
        
        base_name = os.path.basename(npy_file).replace('.npy', '')
        
        # Plot raw confusion matrix
        raw_output = os.path.join(output_dir, f"{base_name}.png")
        plot_confusion_matrix(
            cm,
            labels,
            title="Confusion Matrix",
            save_path=raw_output,
            normalize=False
        )
        print(f"Saved confusion matrix to: {raw_output}")
        
        # Plot normalized confusion matrix
        norm_output = os.path.join(output_dir, f"{base_name}_normalized.png")
        plot_confusion_matrix(
            cm,
            labels,
            title="Normalized Confusion Matrix",
            save_path=norm_output,
            normalize=True
        )
        print(f"Saved normalized confusion matrix to: {norm_output}")
    else:
        # Legacy: plot all confusion matrices from results directory
        results_dir = "results"
        for run_name in os.listdir(results_dir):
            run_path = os.path.join(results_dir, run_name)
            if os.path.isdir(run_path):
                plot_all_confusion_matrices(run_path)
