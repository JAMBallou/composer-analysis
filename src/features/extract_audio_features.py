"""
extract_audio_features.py
-------------------------
Utility to extract features from the MAESTRO audio files and store them as indexed numpy arrays in ``results/features/audio/``.

UPDATED: Now extracts 3 temporal segments (start, middle, end) per piece to capture musical structure.

Audio feature file structure:
- Mel spectrograms (N_MELS x time frames) stored in:
    - xxx_mel_start.npy (first 60s or proportional segment)
    - xxx_mel_middle.npy (centered 60s or proportional segment)
    - xxx_mel_end.npy (last 60s or proportional segment)
- Auxiliary features stored in:
    - xxx_aux_start.npy (105 values: MFCCs 78 + chroma 24 + rhythm 3)
    - xxx_aux_middle.npy
    - xxx_aux_end.npy

Segmentation strategy:
- For pieces >= 180s: Extract 3 distinct 60s segments (start: 0-60s, middle: centered, end: last 60s)
- For pieces 60-180s: Split proportionally into 3 equal segments, then resample each to 60s
- For pieces < 60s: Rejected from dataset
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
OUTPUT_DIR = REPO_ROOT / "outputs" / "features" / "audio"
LABELS_PATH = REPO_ROOT / "outputs" / "features" / "labels.csv"

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

def load_3_segment_audio(path):
    """
    Load audio and extract 3 temporal segments:
    - Start: 0-60s
    - Middle: centered 60s
    - End: last 60s
    
    For pieces < 180s, split proportionally into 3 equal segments.
    
    Returns:
        tuple: (start_segment, middle_segment, end_segment) or None if too short
    """
    y, sr = librosa.load(path, sr=SR)
    total_samples = int(CLIP_DURATION * SR)
    duration_seconds = len(y) / SR
    
    # Reject pieces shorter than 60s
    if duration_seconds < CLIP_DURATION:
        return None
    
    # Case 1: Long pieces (>= 180s) - extract 3 distinct 60s segments
    if duration_seconds >= 180:
        # Start: 0-60s
        start_segment = y[0:total_samples]
        
        # Middle: centered 60s
        mid_point = len(y) // 2
        middle_start = max(0, mid_point - total_samples // 2)
        middle_segment = y[middle_start:middle_start + total_samples]
        
        # End: last 60s
        end_segment = y[-total_samples:]
        
    # Case 2: Short pieces (60-180s) - split proportionally into 3 equal segments
    else:
        segment_length = len(y) // 3
        
        start_segment = y[0:segment_length]
        middle_segment = y[segment_length:2*segment_length]
        end_segment = y[2*segment_length:]
        
        # Resample each segment to 60s (total_samples) for consistency
        start_segment = librosa.resample(start_segment, orig_sr=len(start_segment), target_sr=total_samples)
        middle_segment = librosa.resample(middle_segment, orig_sr=len(middle_segment), target_sr=total_samples)
        end_segment = librosa.resample(end_segment, orig_sr=len(end_segment), target_sr=total_samples)
    
    return (start_segment, middle_segment, end_segment)

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
        segments = load_3_segment_audio(str(audio_path))
        if segments is None:
            continue
        
        start_segment, middle_segment, end_segment = segments

        # Compute features for each segment
        feats_start = compute_audio_features(start_segment)
        feats_middle = compute_audio_features(middle_segment)
        feats_end = compute_audio_features(end_segment)

        # Save features with 4-digit zero-padded index and segment suffix
        idx_str = f"{file_index:04d}"
        
        # Save mel spectrograms
        np.save(OUTPUT_DIR / f"{idx_str}_mel_start.npy", feats_start["mel"])
        np.save(OUTPUT_DIR / f"{idx_str}_mel_middle.npy", feats_middle["mel"])
        np.save(OUTPUT_DIR / f"{idx_str}_mel_end.npy", feats_end["mel"])
        
        # Save auxiliary features
        np.save(OUTPUT_DIR / f"{idx_str}_aux_start.npy", feats_start["aux"])
        np.save(OUTPUT_DIR / f"{idx_str}_aux_middle.npy", feats_middle["aux"])
        np.save(OUTPUT_DIR / f"{idx_str}_aux_end.npy", feats_end["aux"])

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
