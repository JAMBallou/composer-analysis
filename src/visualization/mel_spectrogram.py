"""
mel_spectrogram.py
-------------------------
Plot a mel-spectrogram saved as a .npy file (as produced by extract_audio_features.py). Used to generate visualizations for tri-fold presentation.

Usage:
python -m src.visualization.mel_spectrogram <path_to_mel>.npy
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_mel_spectrogram(mel, save_path, title=None):
    """Plot and save a mel-spectrogram (dB scale)."""
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 16})
    plt.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    cbar = plt.colorbar(label="dB")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("dB", size=16)
    if title:
        plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Time frames", fontsize=16)
    plt.ylabel("Mel bins", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main(npy_path):
    if not os.path.exists(npy_path):
        print(f"Error: File not found: {npy_path}")
        return 1

    if not npy_path.endswith(".npy"):
        print("Error: File must be a .npy file")
        return 1

    mel = np.load(npy_path)
    mel = np.squeeze(mel)

    if mel.ndim != 2:
        print(f"Error: Expected 2D mel spectrogram, got shape {mel.shape}")
        return 1

    output_dir = os.path.dirname(npy_path) or "."
    base_name = os.path.basename(npy_path).replace(".npy", "")
    save_path = os.path.join(output_dir, f"{base_name}_mel.png")

    plot_mel_spectrogram(mel, save_path, title=base_name)
    print(f"Saved mel spectrogram to: {save_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a mel-spectrogram .npy file and save as PNG."
    )
    parser.add_argument("npy_path", help="Path to mel .npy file")
    args = parser.parse_args()

    raise SystemExit(main(args.npy_path))
