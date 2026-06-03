"""
config.py
---------
Holds global constants for the pipeline.
"""

from pathlib import Path

# Repository & dataset paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MAESTRO_DIR = DATA_DIR / "maestro"
MAESTRO_V3_DIR = MAESTRO_DIR / "maestro-v3.0.0"
METADATA_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
COMPOSER_INFO_CSV = MAESTRO_DIR / "composer_info.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Backwards-compatible string path
DATA_PATH = str(MAESTRO_V3_DIR) + "/"

# Audio / feature constants
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
NUM_CLASSES = 14

# Output subdirectories for features/labels
FEATURES_DIR = OUTPUT_DIR / "features"
AUDIO_FEATURES_DIR = FEATURES_DIR / "audio"
MIDI_FEATURES_DIR = FEATURES_DIR / "midi"
LABELS_CSV = FEATURES_DIR / "labels.csv"
