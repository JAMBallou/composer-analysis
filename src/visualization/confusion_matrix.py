"""
confusion_matrix.py
-------------------------
Visualization script for confusion matrices from composer classification models.
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
    # Legacy: plot all confusion matrices from results directory
    results_dir = "results"
    for run_name in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_name)
        if os.path.isdir(run_path):
            plot_all_confusion_matrices(run_path)
