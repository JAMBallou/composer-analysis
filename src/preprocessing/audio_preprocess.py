"""
audio_preprocess.py
-------------------
Converts MAESTRO audio recordings into normalized Mel spectrograms.
"""

import librosa
import numpy as np

def load_and_convert_audio(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Loads a WAV file and returns its Mel spectrogram.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate of ``audio_path``.
        n_fft (int): Number of FFT components to compute.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of Mel bands to generate.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # TODO: apply normalization, optional trimming
    return mel_db
