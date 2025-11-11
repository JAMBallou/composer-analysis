"""
main.py
-------
Main entry point: runs the end-to-end training pipeline.
"""

from src.preprocessing.load_data import load_maestro_metadata
from src.preprocessing.audio_preprocess import load_and_convert_audio
from src.model.cnn_audio import build_audio_cnn
from src.training.train_audio_baseline import train_model

def main():
    # TODO: split dataset into train/val/test
    metadata = load_maestro_metadata("data/maestro-v3.0.0/maestro-v3.0.0.json")

    # TODO: loop through files, generate spectrograms
    # X_train, y_train, X_val, y_val = ...

    model = build_audio_cnn(input_shape=(128, 431, 1), num_classes=14)
    # TODO: train model with extracted features
    # history = train_model(model, X_train, y_train, X_val, y_val)

    print("✅ Pipeline setup complete — ready to train.")

if __name__ == "__main__":
    main()
