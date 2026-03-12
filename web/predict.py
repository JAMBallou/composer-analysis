"""
predict.py
----------
Module for making composer predictions on new audio/MIDI files.
Handles feature extraction and model inference.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Lazy imports - only load heavy libraries when needed
# This speeds up Flask startup significantly
_tf = None
_librosa = None
_mido = None

def _get_tf():
    """Lazy load TensorFlow."""
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf

def _get_librosa():
    """Lazy load librosa."""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa

def _get_mido():
    """Lazy load mido."""
    global _mido
    if _mido is None:
        import mido
        _mido = mido
    return _mido

# Add src to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.advanced_midi_features import (
    compute_pc_transition_matrix,
    compute_key_features,
    compute_intervallic_profile,
    compute_dynamic_features,
)

# Audio processing parameters (match training)
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CLIP_DURATION = 60.0

# Composer names for models (trial 3a subset - 5 composers)
COMPOSERS_T3A = [
    "Frédéric Chopin",
    "Franz Schubert", 
    "Ludwig van Beethoven",
    "Johann Sebastian Bach",
    "Franz Liszt"
]

# All 12 composers (full trial 3b)
COMPOSERS_T3B = [
    "Frédéric Chopin",
    "Franz Schubert",
    "Ludwig van Beethoven",
    "Johann Sebastian Bach",
    "Franz Liszt",
    "Sergei Rachmaninoff",
    "Robert Schumann",
    "Claude Debussy",
    "Joseph Haydn",
    "Wolfgang Amadeus Mozart",
    "Alexander Scriabin",
    "Domenico Scarlatti"
]


def _adapt_vector_dim(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Adapt 1D feature vector to target width by exact, truncate, or zero-pad."""
    current_dim = int(vector.shape[0])
    if current_dim == target_dim:
        return vector
    if current_dim > target_dim:
        return vector[:target_dim]

    padded = np.zeros(target_dim, dtype=vector.dtype)
    padded[:current_dim] = vector
    return padded


def _adapt_stack_dim(stack: np.ndarray, target_dim: int) -> np.ndarray:
    """Adapt 2D feature stack (segments, dim) to target width."""
    current_dim = int(stack.shape[1])
    if current_dim == target_dim:
        return stack
    if current_dim > target_dim:
        return stack[:, :target_dim]

    padded = np.zeros((stack.shape[0], target_dim), dtype=stack.dtype)
    padded[:, :current_dim] = stack
    return padded

ALLOWED_MODEL_RELATIVE_PATHS = [
    "outputs/models/trial1_20260306_061159/trial1_fold4_20260306_083220.keras",
    "outputs/models/trial2_20260205_000000/trial2_fold3_20260205_212913.keras",
    "outputs/models/trial3a_20260206_000000/trial3a_fold1_20260206_125145.keras",
    "outputs/models/trial3b_20260307_055047/trial3b_fold5_20260308_120855.keras",
]


def load_3_segment_audio_for_inference(path):
    """Load audio and extract 3 temporal segments for inference."""
    librosa = _get_librosa()

    y, _ = librosa.load(path, sr=SR)
    total_samples = int(CLIP_DURATION * SR)
    duration_seconds = len(y) / SR

    if duration_seconds < CLIP_DURATION:
        return None

    if duration_seconds >= 180:
        start_segment = y[0:total_samples]
        mid_point = len(y) // 2
        middle_start = max(0, mid_point - total_samples // 2)
        middle_segment = y[middle_start:middle_start + total_samples]
        end_segment = y[-total_samples:]
    else:
        segment_length = len(y) // 3
        start_segment = y[0:segment_length]
        middle_segment = y[segment_length:2 * segment_length]
        end_segment = y[2 * segment_length:]

        start_segment = librosa.resample(start_segment, orig_sr=max(1, len(start_segment)), target_sr=total_samples)
        middle_segment = librosa.resample(middle_segment, orig_sr=max(1, len(middle_segment)), target_sr=total_samples)
        end_segment = librosa.resample(end_segment, orig_sr=max(1, len(end_segment)), target_sr=total_samples)

    return (start_segment, middle_segment, end_segment)


def compute_audio_features_for_inference(y):
    """Compute mel spectrogram + 105-dim auxiliary features for one segment."""
    librosa = _get_librosa()

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, delta, delta2])

    mfcc_stats = np.concatenate([
        mfcc_all.mean(axis=1),
        mfcc_all.std(axis=1),
    ])

    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    chroma_stats = np.concatenate([
        chroma.mean(axis=1),
        chroma.std(axis=1),
    ])

    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR)
    tempo_val = float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)

    rhythm_features = np.array([
        tempo_val,
        float(onset_env.mean()),
        float(onset_env.std()),
    ], dtype=np.float32)

    aux_features = np.concatenate([
        mfcc_stats,
        chroma_stats,
        rhythm_features,
    ]).astype(np.float32)

    return {"mel": mel_db, "aux": aux_features}


def get_segment_boundaries(total_duration, segment_duration=60.0):
    if total_duration < segment_duration:
        return None

    if total_duration >= 180:
        start_segment = (0.0, segment_duration)
        mid_point = total_duration / 2
        middle_segment = (mid_point - segment_duration / 2, mid_point + segment_duration / 2)
        end_segment = (total_duration - segment_duration, total_duration)
    else:
        segment_length = total_duration / 3
        start_segment = (0.0, segment_length)
        middle_segment = (segment_length, 2 * segment_length)
        end_segment = (2 * segment_length, total_duration)

    return [start_segment, middle_segment, end_segment]


def filter_notes_by_time_range(all_notes, control_changes, time_start, time_end):
    filtered_notes = []
    for pitch, start, end, velocity, dur in all_notes:
        if start < time_end and end > time_start:
            clipped_start = max(start, time_start) - time_start
            clipped_end = min(end, time_end) - time_start
            clipped_dur = clipped_end - clipped_start
            if clipped_dur > 0:
                filtered_notes.append((pitch, clipped_start, clipped_end, velocity, clipped_dur))

    filtered_control_changes = []
    for time, value in control_changes:
        if time_start <= time <= time_end:
            filtered_control_changes.append((time - time_start, value))

    return filtered_notes, filtered_control_changes


def extract_notes_from_midi_for_inference(midi_path):
    mido = _get_mido()

    try:
        midi_file = mido.MidiFile(midi_path)
    except Exception:
        return None

    tempo = 500000
    ticks_per_beat = midi_file.ticks_per_beat

    all_notes = []
    control_changes = []

    for track in midi_file.tracks:
        current_time = 0.0
        active_notes = {}

        for msg in track:
            current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                if msg.note not in active_notes:
                    active_notes[msg.note] = (current_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_time, velocity = active_notes[msg.note]
                    duration = current_time - start_time
                    all_notes.append((msg.note, start_time, current_time, velocity, duration))
                    del active_notes[msg.note]
            elif msg.type == 'control_change' and msg.control == 64:
                control_changes.append((current_time, msg.value))

    if not all_notes:
        return None

    return all_notes, control_changes


def compute_midi_features_from_notes(all_notes, control_changes, duration):
    if not all_notes:
        return None

    pc_histogram = np.zeros(12, dtype=np.float32)
    note_durations = []
    note_pitches = []

    for pitch, start, end, velocity, dur in all_notes:
        pitch_class = pitch % 12
        pc_histogram[pitch_class] += 1
        note_durations.append(dur)
        note_pitches.append(pitch)

    pc_histogram = pc_histogram / (len(all_notes) + 1e-8)

    note_durations = np.array(note_durations, dtype=np.float32)
    note_pitches = np.array(note_pitches, dtype=np.float32)

    mean_duration = np.mean(note_durations) if len(note_durations) > 0 else 0.0
    std_duration = np.std(note_durations) if len(note_durations) > 0 else 0.0

    staccato_count = np.sum(note_durations < 0.1)
    sustained_count = np.sum(note_durations > 0.5)
    staccato_ratio = staccato_count / (len(all_notes) + 1e-8)
    sustained_ratio = sustained_count / (len(all_notes) + 1e-8)

    articulation_ratio = min(len(all_notes) / (duration + 1e-8) / 10.0, 1.0)

    sorted_notes = sorted(all_notes, key=lambda n: n[1])
    legato_count = 0
    for i in range(1, len(sorted_notes)):
        if sorted_notes[i][1] < sorted_notes[i - 1][2]:
            legato_count += 1
    legato_ratio = legato_count / max(len(all_notes) - 1, 1)

    note_stats = np.array([
        float(mean_duration),
        float(std_duration),
        float(staccato_ratio),
        float(sustained_ratio),
        float(articulation_ratio),
        float(legato_ratio),
    ], dtype=np.float32)

    mean_pitch = np.mean(note_pitches)
    std_pitch = np.std(note_pitches)
    bass_fraction = np.sum(note_pitches < 54) / (len(note_pitches) + 1e-8)
    middle_fraction = np.sum((note_pitches >= 54) & (note_pitches < 72)) / (len(note_pitches) + 1e-8)
    treble_fraction = np.sum(note_pitches >= 72) / (len(note_pitches) + 1e-8)

    register_usage = np.array([
        float(mean_pitch),
        float(std_pitch),
        float(bass_fraction),
        float(middle_fraction),
        float(treble_fraction),
    ], dtype=np.float32)

    mean_notes_per_second = len(all_notes) / (duration + 1e-8)
    time_points = []
    for pitch, start, end, velocity, dur in all_notes:
        time_points.append((start, 1))
        time_points.append((end, -1))

    time_points.sort()
    simultaneous_notes = []
    current_simultaneous = 0
    for _, delta in time_points:
        current_simultaneous += delta
        simultaneous_notes.append(current_simultaneous)

    mean_simultaneous = np.mean(simultaneous_notes) if simultaneous_notes else 0.0
    max_simultaneous = max(simultaneous_notes) if simultaneous_notes else 0.0

    note_density = np.array([
        float(mean_notes_per_second),
        float(mean_simultaneous),
        float(max_simultaneous),
    ], dtype=np.float32)

    if control_changes:
        sustain_values = [val for _, val in control_changes]
        avg_sustain_value = np.mean(sustain_values)
        pedal_on_count = sum(1 for _, val in control_changes if val > 63)
        pedal_on_fraction = pedal_on_count / (len(control_changes) + 1e-8)
        pedal_variance = np.var(sustain_values)
    else:
        avg_sustain_value = 0.0
        pedal_on_fraction = 0.0
        pedal_variance = 0.0

    pedal_usage = np.array([
        float(avg_sustain_value),
        float(pedal_on_fraction),
        float(pedal_variance),
    ], dtype=np.float32)

    basic_features = np.concatenate([
        pc_histogram,
        note_stats,
        register_usage,
        note_density,
        pedal_usage,
    ]).astype(np.float32)

    harmonic_features = compute_pc_transition_matrix(all_notes)
    key_features = compute_key_features(all_notes, pc_histogram)
    intervallic_features = compute_intervallic_profile(all_notes)
    dynamic_features = compute_dynamic_features(all_notes, control_changes, duration)

    full_features = np.concatenate([
        basic_features,
        harmonic_features,
        key_features,
        intervallic_features,
        dynamic_features,
    ]).astype(np.float32)

    return full_features


def compute_midi_features_3_segments_for_inference(midi_path):
    result = extract_notes_from_midi_for_inference(midi_path)
    if result is None:
        return None

    all_notes, control_changes = result
    if not all_notes:
        return None

    max_time = max(note[2] for note in all_notes)
    total_duration = max_time if max_time > 0 else 1.0

    segments = get_segment_boundaries(total_duration)
    if segments is None:
        return None

    features_list = []
    for time_start, time_end in segments:
        segment_notes, segment_controls = filter_notes_by_time_range(
            all_notes, control_changes, time_start, time_end
        )
        segment_duration = time_end - time_start

        features = compute_midi_features_from_notes(
            segment_notes, segment_controls, segment_duration
        )

        if features is None:
            features = np.zeros(64, dtype=np.float32)

        features_list.append(features)

    return tuple(features_list)


def extract_audio_features_for_prediction(audio_path):
    """
    Extract audio features from a file for prediction.
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
    
    Returns:
        dict: Contains 'mel' (spectrogram) and 'aux' (auxiliary features)
              or None if extraction fails
    """
    try:
        # Load 3 segments
        segments = load_3_segment_audio_for_inference(str(audio_path))
        
        if segments is None:
            return None
        
        start_segment, middle_segment, end_segment = segments
        
        # Compute features for each segment
        feats_start = compute_audio_features_for_inference(start_segment)
        feats_middle = compute_audio_features_for_inference(middle_segment)
        feats_end = compute_audio_features_for_inference(end_segment)
        
        # Stack mel spectrograms: shape (3, 128, time_frames)
        mel_stack = np.stack([
            feats_start["mel"],
            feats_middle["mel"],
            feats_end["mel"]
        ], axis=0)
        
        # Stack auxiliary features: shape (3, 105)
        aux_stack = np.stack([
            feats_start["aux"],
            feats_middle["aux"],
            feats_end["aux"]
        ], axis=0)
        
        return {
            "mel": mel_stack,
            "aux": aux_stack
        }
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None


def extract_midi_features_for_prediction(midi_path):
    """
    Extract MIDI features from a file for prediction.
    
    Args:
        midi_path: Path to MIDI file (.mid, .midi)
    
    Returns:
        numpy array of shape (3, 64) or None if extraction fails
    """
    try:
        features_tuple = compute_midi_features_3_segments_for_inference(str(midi_path))
        
        if features_tuple is None:
            return None
        
        features_start, features_middle, features_end = features_tuple
        
        # Stack features: shape (3, 64)
        midi_stack = np.stack([
            features_start,
            features_middle,
            features_end
        ], axis=0)
        
        return midi_stack
        
    except Exception as e:
        print(f"Error extracting MIDI features: {e}")
        return None


def get_model_info(model_path):
    """
    Extract trial information from model path to determine which composers it predicts.
    
    Args:
        model_path: Path to .keras model file
    
    Returns:
        dict with 'composers' list and 'trial' name
    """
    model_name = Path(model_path).name
    
    # Detect trial type from filename
    if "trial3a" in model_name or "t3a" in model_name:
        return {
            "trial": "trial3a",
            "composers": COMPOSERS_T3A,
            "description": "5 composers (>80 works each)"
        }
    elif "trial3b" in model_name or "t3b" in model_name:
        return {
            "trial": "trial3b",
            "composers": COMPOSERS_T3B,
            "description": "12 composers (full subset)"
        }
    elif "trial2" in model_name or "t2" in model_name:
        return {
            "trial": "trial2",
            "composers": ["Frédéric Chopin", "Franz Schubert"],
            "description": "2 similar composers (Romantic period)"
        }
    elif "trial1" in model_name or "t1" in model_name:
        return {
            "trial": "trial1",
            "composers": ["Johann Sebastian Bach", "Frédéric Chopin"],
            "description": "2 contrasting composers"
        }
    else:
        # Default to trial3a
        return {
            "trial": "unknown",
            "composers": COMPOSERS_T3A,
            "description": "Unknown trial type"
        }


def predict_composer(audio_path=None, midi_path=None, model_path=None):
    """
    Predict composer from audio and/or MIDI file.
    
    Args:
        audio_path: Path to audio file (optional)
        midi_path: Path to MIDI file (optional)
        model_path: Path to trained .keras model
    
    Returns:
        dict: {
            "composer": predicted composer name,
            "confidence": confidence score,
            "probabilities": dict of all composer probabilities,
            "model_info": dict with model metadata
        }
    """
    if model_path is None:
        raise ValueError("model_path is required")
    
    if audio_path is None and midi_path is None:
        raise ValueError("At least one of audio_path or midi_path is required")
    
    # Load model for inference only (skip training config like custom losses)
    tf = _get_tf()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    # Get model info
    model_info = get_model_info(model_path)
    composers = model_info["composers"]
    
    # Extract features
    audio_features = None
    midi_features = None
    
    if audio_path:
        audio_features = extract_audio_features_for_prediction(audio_path)
        if audio_features is None:
            raise ValueError("Failed to extract audio features (file may be too short or invalid)")
    
    if midi_path:
        midi_features = extract_midi_features_for_prediction(midi_path)
        if midi_features is None:
            raise ValueError("Failed to extract MIDI features (file may be invalid)")
    
    # Prepare input based on what's available
    # Check model input signature to determine expected inputs
    input_shapes = [inp.shape for inp in model.inputs]
    
    # If only audio is provided, we need to create dummy MIDI features
    if audio_path and not midi_path:
        # Create zero-padded MIDI features
        midi_features = np.zeros((3, 64), dtype=np.float32)
    
    # If only MIDI is provided, we need to create dummy audio features
    if midi_path and not audio_path:
        # Create zero-padded audio features
        # Need to match expected audio shape from model
        # Typical shape: (128, time_frames)
        time_frames = 2580  # Default based on 60s @ SR=22050, hop=512
        audio_features = {
            "mel": np.zeros((3, 128, time_frames), dtype=np.float32),
            "aux": np.zeros((3, 105), dtype=np.float32)
        }
    
    # Prepare model inputs
    # Check if model is temporal (expects 3 segments) or standard
    try:
        # For temporal models with 6 inputs: 
        # [spec_start, spec_middle, spec_end, aux_start, aux_middle, aux_end, 
        #  midi_start, midi_middle, midi_end]
        
        if len(input_shapes) >= 6:  # Temporal model
            # Add batch dimension and ensure correct shape
            mel_stack = audio_features["mel"]  # (3, 128, time_frames)
            aux_stack = audio_features["aux"]  # (3, 105)
            midi_stack = midi_features  # (3, 64)

            # Some temporal models expect combined numerical features
            # (aux + midi => 169) in num_feat_* inputs.
            num_feat_stack = np.concatenate([aux_stack, midi_stack], axis=1)
            
            # Add channel dimension to mel spectrograms
            mel_stack = mel_stack[..., np.newaxis]  # (3, 128, time_frames, 1)
            
            # Normalize
            mel_stack = (mel_stack - mel_stack.mean()) / (mel_stack.std() + 1e-8)
            aux_stack = (aux_stack - aux_stack.mean()) / (aux_stack.std() + 1e-8)
            midi_stack = (midi_stack - midi_stack.mean()) / (midi_stack.std() + 1e-8)
            num_feat_stack = (num_feat_stack - num_feat_stack.mean()) / (num_feat_stack.std() + 1e-8)

            # Infer expected numerical feature width from model signature
            # Input 3 is first numerical branch in temporal models.
            expected_num_dim = input_shapes[3][-1]
            expected_num_dim = int(expected_num_dim) if expected_num_dim is not None else num_feat_stack.shape[1]

            if expected_num_dim == num_feat_stack.shape[1]:
                temporal_num_stack = num_feat_stack
            elif expected_num_dim == aux_stack.shape[1]:
                temporal_num_stack = aux_stack
            elif expected_num_dim == midi_stack.shape[1]:
                temporal_num_stack = midi_stack
            else:
                # Fallback for legacy widths (e.g., 29): prefer MIDI basis, then adapt
                temporal_num_stack = _adapt_stack_dim(midi_stack, expected_num_dim)
            
            # Prepare inputs
            inputs = [
                np.expand_dims(mel_stack[0], 0),   # spec_start
                np.expand_dims(mel_stack[1], 0),   # spec_middle
                np.expand_dims(mel_stack[2], 0),   # spec_end
                np.expand_dims(temporal_num_stack[0], 0),   # num_feat_start/aux_start
                np.expand_dims(temporal_num_stack[1], 0),   # num_feat_middle/aux_middle
                np.expand_dims(temporal_num_stack[2], 0),   # num_feat_end/aux_end
            ]
            
            # Add MIDI inputs if model expects them
            if len(input_shapes) >= 9:
                inputs.extend([
                    np.expand_dims(midi_stack[0], 0),  # midi_start
                    np.expand_dims(midi_stack[1], 0),  # midi_middle
                    np.expand_dims(midi_stack[2], 0),  # midi_end
                ])
        
        else:  # Standard multimodal model (2 inputs)
            # Use middle segment or aggregate
            mel = audio_features["mel"][1]  # Use middle segment
            mel = mel[..., np.newaxis]  # Add channel dimension
            aux = audio_features["aux"][1]
            midi = midi_features[1]  # Use middle segment
            combined = np.concatenate([aux, midi], axis=0)
            
            # Normalize
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
            aux = (aux - aux.mean()) / (aux.std() + 1e-8)
            midi = (midi - midi.mean()) / (midi.std() + 1e-8)
            combined = (combined - combined.mean()) / (combined.std() + 1e-8)

            # Choose feature vector width expected by second input
            expected_feat_dim = input_shapes[1][-1]
            expected_feat_dim = int(expected_feat_dim) if expected_feat_dim is not None else combined.shape[0]

            if expected_feat_dim == combined.shape[0]:
                second_input = combined
            elif expected_feat_dim == aux.shape[0]:
                second_input = aux
            elif expected_feat_dim == midi.shape[0]:
                second_input = midi
            else:
                # Fallback for legacy widths (e.g., 29): prefer MIDI basis, then adapt
                second_input = _adapt_vector_dim(midi, expected_feat_dim)
            
            inputs = [
                np.expand_dims(mel, 0),
                np.expand_dims(second_input, 0)
            ]
        
        # Make prediction
        predictions = model.predict(inputs, verbose=0)
        
        # Get predicted class and confidence
        prediction_vector = predictions[0]
        num_output_classes = int(prediction_vector.shape[0])

        # Align display labels to actual model output width
        if len(composers) >= num_output_classes:
            display_labels = composers[:num_output_classes]
        else:
            display_labels = composers + [
                f"Class {i + 1}" for i in range(len(composers), num_output_classes)
            ]

        predicted_idx = int(np.argmax(prediction_vector))
        confidence = float(prediction_vector[predicted_idx])
        
        # Build probabilities dictionary
        probabilities = {
            display_labels[i]: float(prediction_vector[i])
            for i in range(num_output_classes)
        }
        
        return {
            "composer": display_labels[predicted_idx],
            "confidence": confidence,
            "probabilities": probabilities,
            "model_info": {
                **model_info,
                "model_output_classes": num_output_classes,
                "label_count": len(display_labels),
            }
        }
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")


def get_available_models(models_dir="outputs/models"):
    """
    List available trained models.
    
    Args:
        models_dir: Path to models directory
    
    Returns:
        list of dicts with model information
    """
    models = []

    # Enforce explicit allowlist and preserve this exact order.
    for rel_path in ALLOWED_MODEL_RELATIVE_PATHS:
        model_file = Path(REPO_ROOT) / Path(rel_path)
        if not model_file.exists():
            continue

        info = get_model_info(str(model_file))
        models.append({
            "path": str(model_file),
            "name": model_file.name,
            "trial": info["trial"],
            "description": info["description"],
            "composers": info["composers"]
        })

    return models


if __name__ == "__main__":
    # Test functionality
    print("Available models:")
    models = get_available_models()
    for i, model in enumerate(models):
        print(f"{i+1}. {model['name']} - {model['description']}")
