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
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
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

def load_confusion_matrix(run_dir):
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
        raise FileNotFoundError("No confusion matrix found.")

    return cm, labels

def plot_all_confusion_matrices(
    results_dir="results",
    output_dir="results/confusion_matrices"
):
    os.makedirs(output_dir, exist_ok=True)

    for run_name in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, run_name)

        if not os.path.isdir(run_dir):
            continue

        try:
            cm, labels = load_confusion_matrix(run_dir)
        except FileNotFoundError:
            continue

        base_name = run_name.replace("/", "_")

        # Raw confusion matrix
        plot_confusion_matrix(
            cm,
            labels,
            title=f"Confusion Matrix — {run_name}",
            save_path=os.path.join(
                output_dir, f"{base_name}_raw.png"
            ),
            normalize=False
        )

        # Normalized confusion matrix
        plot_confusion_matrix(
            cm,
            labels,
            title=f"Normalized Confusion Matrix — {run_name}",
            save_path=os.path.join(
                output_dir, f"{base_name}_normalized.png"
            ),
            normalize=True
        )

        print(f"Saved confusion matrices for {run_name}")

if __name__ == "__main__":
    plot_all_confusion_matrices(
        results_dir="results",
        output_dir="results/confusion_matrices"
    )
