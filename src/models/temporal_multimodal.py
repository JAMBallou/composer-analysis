"""
temporal_multimodal.py
-------------------------
Temporal multimodal model for composer classification using 3-segment architecture.

Architecture:
    1. Three spectrograms (start, middle, end) → 3 CNNs (shared weights) → 3 embeddings
    2. Three sets of numerical features (audio aux + MIDI) for each segment
    3. Combine: [(spec_emb_1, num_feat_1), (spec_emb_2, num_feat_2), (spec_emb_3, num_feat_3)]
    4. BiLSTM processes the temporal sequence
    5. BiLSTM output → Dense classifier
"""

import tensorflow as tf
from keras import layers, models

from .cnn import build_cnn_audio_model
from .mlp import build_feature_mlp


def build_temporal_multimodal_model(
    mel_bins: int,
    time_frames: int,
    num_aux_features: int,
    num_midi_features: int,
    num_classes: int,
    cnn_embedding_dim: int = 256,
    numerical_embedding_dim: int = 128,
    lstm_units: int = 128,
    dropout_rate: float = 0.4,
    use_shared_cnn: bool = True
) -> tf.keras.Model:
    """
    Temporal multimodal composer classification model.
    
    Processes 3 temporal segments (start, middle, end) of audio spectrograms
    and numerical features through a BiLSTM for temporal modeling.
    
    Args:
        mel_bins: Number of mel frequency bins (e.g., 128)
        time_frames: Number of time frames in each spectrogram segment
        num_aux_features: Length of auxiliary audio feature vector per segment
        num_midi_features: Length of MIDI feature vector per segment
        num_classes: Number of output composer classes
        cnn_embedding_dim: Size of CNN spectrogram embedding
        numerical_embedding_dim: Size of numerical feature embedding
        lstm_units: Number of units in BiLSTM layer
        dropout_rate: Dropout rate for regularization
        use_shared_cnn: Whether to use shared CNN weights for all 3 segments
    
    Returns:
        tf.keras.Model: Compiled temporal multimodal model
    """
    
    # ===== Inputs =====
    # Spectrograms
    spec_start = layers.Input(shape=(mel_bins, time_frames, 1), name="spec_start")
    spec_middle = layers.Input(shape=(mel_bins, time_frames, 1), name="spec_middle")
    spec_end = layers.Input(shape=(mel_bins, time_frames, 1), name="spec_end")
    
    # Numerical features (aux audio + MIDI)
    num_feat_start = layers.Input(shape=(num_aux_features + num_midi_features,), name="num_feat_start")
    num_feat_middle = layers.Input(shape=(num_aux_features + num_midi_features,), name="num_feat_middle")
    num_feat_end = layers.Input(shape=(num_aux_features + num_midi_features,), name="num_feat_end")
    
    # ===== CNN Branch: Process Spectrograms =====
    if use_shared_cnn:
        # Shared CNN for all 3 segments
        shared_cnn = build_cnn_audio_model(
            mel_bins=mel_bins,
            time_frames=time_frames,
            num_classes=cnn_embedding_dim,
            output_activation=None
        )
        
        # Remove classification head, use as feature extractor
        shared_cnn_base = models.Model(
            inputs=shared_cnn.input,
            outputs=shared_cnn.layers[-2].output,  # Before final Dense layer
            name="shared_cnn_base"
        )
        
        spec_emb_start = shared_cnn_base(spec_start)
        spec_emb_middle = shared_cnn_base(spec_middle)
        spec_emb_end = shared_cnn_base(spec_end)
    else:
        # Separate CNNs for each segment
        cnn_start = build_cnn_audio_model(mel_bins, time_frames, cnn_embedding_dim, output_activation=None)
        cnn_middle = build_cnn_audio_model(mel_bins, time_frames, cnn_embedding_dim, output_activation=None)
        cnn_end = build_cnn_audio_model(mel_bins, time_frames, cnn_embedding_dim, output_activation=None)
        
        spec_emb_start = cnn_start(spec_start)
        spec_emb_middle = cnn_middle(spec_middle)
        spec_emb_end = cnn_end(spec_end)
    
    # Project CNN embeddings to consistent dimension
    spec_projection = layers.Dense(cnn_embedding_dim, activation="relu", name="spec_projection")
    spec_emb_start = spec_projection(spec_emb_start)
    spec_emb_middle = spec_projection(spec_emb_middle)
    spec_emb_end = spec_projection(spec_emb_end)
    
    # ===== Numerical Feature Branch =====
    # Shared MLP for numerical features
    num_total_features = num_aux_features + num_midi_features
    
    shared_mlp = models.Sequential([
        layers.Dense(numerical_embedding_dim * 2, activation="relu", name="num_dense1"),
        layers.BatchNormalization(name="num_bn1"),
        layers.Dropout(dropout_rate, name="num_dropout1"),
        layers.Dense(numerical_embedding_dim, activation="relu", name="num_dense2"),
        layers.BatchNormalization(name="num_bn2"),
    ], name="shared_numerical_mlp")
    
    num_emb_start = shared_mlp(num_feat_start)
    num_emb_middle = shared_mlp(num_feat_middle)
    num_emb_end = shared_mlp(num_feat_end)
    
    # ===== Combine Modalities per Time Step =====
    # Concatenate spectrogram embedding + numerical embedding for each segment
    combined_start = layers.Concatenate(name="combine_start")([spec_emb_start, num_emb_start])
    combined_middle = layers.Concatenate(name="combine_middle")([spec_emb_middle, num_emb_middle])
    combined_end = layers.Concatenate(name="combine_end")([spec_emb_end, num_emb_end])
    
    # Stack into temporal sequence: shape (batch, 3, feature_dim)
    # Expand dims to add time dimension
    combined_start = layers.Reshape((1, cnn_embedding_dim + numerical_embedding_dim))(combined_start)
    combined_middle = layers.Reshape((1, cnn_embedding_dim + numerical_embedding_dim))(combined_middle)
    combined_end = layers.Reshape((1, cnn_embedding_dim + numerical_embedding_dim))(combined_end)
    
    # Concatenate along time axis
    temporal_sequence = layers.Concatenate(axis=1, name="temporal_sequence")([
        combined_start,
        combined_middle,
        combined_end
    ])
    
    # ===== Temporal Modeling with BiLSTM =====
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate),
        name="bilstm"
    )(temporal_sequence)
    
    x = layers.BatchNormalization(name="bilstm_bn")(x)
    x = layers.Dropout(dropout_rate, name="bilstm_dropout")(x)
    
    # ===== Classification Head =====
    x = layers.Dense(256, activation="relu", name="classifier_dense1")(x)
    x = layers.BatchNormalization(name="classifier_bn1")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout1")(x)
    
    x = layers.Dense(128, activation="relu", name="classifier_dense2")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout2")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="composer_output")(x)
    
    # ===== Build Model =====
    model = models.Model(
        inputs=[
            spec_start, spec_middle, spec_end,
            num_feat_start, num_feat_middle, num_feat_end
        ],
        outputs=outputs,
        name="temporal_multimodal_classifier"
    )
    
    return model


def build_temporal_multimodal_model_simple(
    mel_bins: int,
    time_frames: int,
    num_aux_features: int,
    num_midi_features: int,
    num_classes: int,
    lstm_units: int = 128,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Simplified temporal model with lighter CNN architecture.
    Useful for faster training/experimentation.
    """
    # ===== Inputs =====
    spec_start = layers.Input(shape=(mel_bins, time_frames, 1), name="spec_start")
    spec_middle = layers.Input(shape=(mel_bins, time_frames, 1), name="spec_middle")
    spec_end = layers.Input(shape=(mel_bins, time_frames, 1), name="spec_end")
    
    num_feat_start = layers.Input(shape=(num_aux_features + num_midi_features,), name="num_feat_start")
    num_feat_middle = layers.Input(shape=(num_aux_features + num_midi_features,), name="num_feat_middle")
    num_feat_end = layers.Input(shape=(num_aux_features + num_midi_features,), name="num_feat_end")
    
    # ===== Lightweight CNN =====
    def simple_cnn(x):
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        return x
    
    spec_emb_start = simple_cnn(spec_start)
    spec_emb_middle = simple_cnn(spec_middle)
    spec_emb_end = simple_cnn(spec_end)
    
    # ===== Numerical Features =====
    def simple_mlp(x):
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        return x
    
    num_emb_start = simple_mlp(num_feat_start)
    num_emb_middle = simple_mlp(num_feat_middle)
    num_emb_end = simple_mlp(num_feat_end)
    
    # ===== Combine and Sequence =====
    combined_start = layers.Concatenate()([spec_emb_start, num_emb_start])
    combined_middle = layers.Concatenate()([spec_emb_middle, num_emb_middle])
    combined_end = layers.Concatenate()([spec_emb_end, num_emb_end])
    
    combined_start = layers.Reshape((1, -1))(combined_start)
    combined_middle = layers.Reshape((1, -1))(combined_middle)
    combined_end = layers.Reshape((1, -1))(combined_end)
    
    temporal_sequence = layers.Concatenate(axis=1)([combined_start, combined_middle, combined_end])
    
    # ===== BiLSTM =====
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(temporal_sequence)
    x = layers.Dropout(dropout_rate)(x)
    
    # ===== Classifier =====
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(
        inputs=[spec_start, spec_middle, spec_end, num_feat_start, num_feat_middle, num_feat_end],
        outputs=outputs,
        name="temporal_multimodal_simple"
    )
    
    return model
