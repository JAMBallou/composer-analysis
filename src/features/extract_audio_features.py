"""
extract_audio_features.py
-------------------------
Utility to extract features from the MAESTRO audio files and store them as indexed numpy arrays in ``results/features/audio/``.

Audio feature file structure (xxx_mel.npy, xxx_aux.npy):
- Mel spectrogram (N_MELS x time frames) stored in xxx_mel.npy
- Auxiliary features (MFCCs, chroma, rhythm) stored in xxx_aux.npy:
    - MFCCs: mean and std of 13 MFCCs + deltas + delta-deltas (78 values)
    - Chroma: mean and std of 12 chroma features (24 values)
    - Rhythm: estimated tempo; onset strength mean and std (3 values)
    - Total auxiliary feature vector length: 105
"""

import os
import sys
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Compute paths relative to repo root (project dir)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "maestro" / "data"
METADATA_PATH = REPO_ROOT / "data" / "maestro" / "maestro-v3.0.0.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "features" / "audio"
LABELS_PATH = REPO_ROOT / "results" / "features" / "labels.csv"

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CLIP_DURATION = 60.0  # seconds

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Validate paths exist
if not DATASET_DIR.exists():
    print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
    sys.exit(1)
if not METADATA_PATH.exists():
    print(f"ERROR: Metadata CSV not found: {METADATA_PATH}")
    sys.exit(1)


# ================== HELPERS ==================

def load_60s_audio(path):
    y, sr = librosa.load(path, sr=SR)
    total_samples = int(CLIP_DURATION * SR)

    if len(y) < total_samples:
        return None

    if len(y) > int(90 * SR):
        start = int(30 * SR)
    else:
        start = max(0, len(y) // 2 - total_samples // 2)

    return y[start:start + total_samples]

def compute_audio_features(y):
    """
    Audio feature file structure (xxx_mel.npy, xxx_aux.npy):
    - Mel spectrogram (N_MELS x time frames) stored in xxx_mel.npy
    - Auxiliary features (MFCCs, chroma, rhythm) stored in xxx_aux.npy:
        - MFCCs: mean and std of 13 MFCCs + deltas + delta-deltas (78 values)
        - Chroma: mean and std of 12 chroma features (24 values)
        - Rhythm: estimated tempo; onset strength mean and std (3 values)
        - Total auxiliary feature vector length: 105

    Args:
        y (np.ndarray): Audio time series  
    
    Returns:
        dict: {"mel": mel_spectrogram, "aux": auxiliary_features}
    """

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # ===== AUXILIARY FEATURES =====
    # MFCCs (mean + std)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, delta, delta2])

    mfcc_stats = np.concatenate([
        mfcc_all.mean(axis=1),
        mfcc_all.std(axis=1)
    ])

    # Chroma stats
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    chroma_stats = np.concatenate([
        chroma.mean(axis=1),
        chroma.std(axis=1)
    ])

    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR)

    rhythm_features = np.array([
        float(tempo),
        float(onset_env.mean()),
        float(onset_env.std())
    ], dtype=np.float32)

    # Combine auxiliary features
    aux_features = np.concatenate([
        mfcc_stats,         # 78
        chroma_stats,       # 24
        rhythm_features     # 3
    ]).astype(np.float32)
    
    return {
        "mel": mel_db,
        "aux": aux_features
    }


# ================== MAIN ==================

print(f"Loading metadata from: {METADATA_PATH}")
metadata = pd.read_csv(METADATA_PATH)
print(f"Loaded {len(metadata)} files from CSV")
print(f"Processing files...")

label_rows = []
pbar = tqdm(total=len(metadata), desc="Processing")
file_index = 0

for idx, row in metadata.iterrows():
    pbar.update(1)
    audio_path = DATASET_DIR / row["audio_filename"]

    # Skip if file doesn't exist
    if not audio_path.exists():
        continue

    try:
        y = load_60s_audio(str(audio_path))
        if y is None:
            continue

        feats = compute_audio_features(y)

        # Save features with 4-digit zero-padded index
        idx_str = f"{file_index:04d}"
        np.save(OUTPUT_DIR / f"{idx_str}_mel.npy", feats["mel"])
        np.save(OUTPUT_DIR / f"{idx_str}_aux.npy", feats["aux"])

        label_rows.append({
            "id": idx_str,
            "audio_filename": row["audio_filename"],
            "composer": row.get("canonical_composer", "unknown"),
            "period": row.get("period", "unknown")
        })
        file_index += 1
    except Exception as e:
        print(f"ERROR processing {audio_path}: {e}")
        continue

pbar.close()

# Save labels
print(f"Saving {len(label_rows)} labels to {LABELS_PATH}")
pd.DataFrame(label_rows).to_csv(LABELS_PATH, index=False)
print("Done!")
