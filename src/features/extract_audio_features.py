"""
extract_audio_features.py
-------------------------
Extracts high-level audio descriptors from spectrograms.
"""

import numpy as np
import librosa

def compute_audio_features(y, sr):
    """
    Computes MFCCs, chroma, spectral centroid, etc.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # TODO: flatten or concatenate into single feature vector
    return np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1), spec_centroid.mean(axis=1)])
