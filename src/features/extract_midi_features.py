"""
extract_midi_features.py
-------------------------
Utility to extract features from the MAESTRO MIDI files and store them as indexed numpy arrays in ``results/features/midi/``.

UPDATED: Now extracts 3 temporal segments (start, middle, end) per piece with ADVANCED harmonic & dynamic features.

MIDI feature file structure:
- Feature vectors stored in:
    - xxx_midi_start.npy (64 values from first segment)
    - xxx_midi_middle.npy (64 values from middle segment)
    - xxx_midi_end.npy (64 values from end segment)
    
- Each vector contains:
    - Original Features (29):
        - PC Histogram (12): Note count per pitch class
        - Note Stats (6): Duration & articulation ratios
        - Register Usage (5): Pitch statistics
        - Note Density (3): Temporal density
        - Pedal Usage (3): Sustain pedal
    
    - Advanced Features (35):
        - Harmonic Movement (8): PC transition matrix statistics
        - Key Analysis (5): Key clarity, changes, tonal centroid
        - Intervallic Profile (6): Melodic interval statistics
        - Dynamics (16): Velocity-based features

Total: 64 features per segment

Segmentation strategy:
- Matches audio segmentation (180s threshold)
- For long pieces: 3 distinct 60s segments
- For short pieces: proportional 3-way split
"""

import os
import sys
from pathlib import Path
from utils.config import MAESTRO_DIR, METADATA_CSV, MIDI_FEATURES_DIR, LABELS_CSV
import numpy as np
import pandas as pd
from tqdm import tqdm
import mido

# Import advanced feature functions
from .advanced_midi_features import (
    compute_pc_transition_matrix,
    compute_key_features,
    compute_intervallic_profile,
    compute_dynamic_features
)

# Dataset + output paths from config
METADATA_PATH = METADATA_CSV
DATASET_DIR = MAESTRO_DIR / "data"
OUTPUT_DIR = MIDI_FEATURES_DIR
LABELS_PATH = LABELS_CSV

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Validate paths exist
if not DATASET_DIR.exists():
    print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
    sys.exit(1)
if not METADATA_PATH.exists():
    print(f"ERROR: Metadata CSV not found: {METADATA_PATH}")
    sys.exit(1)


# ================== MIDI FEATURE EXTRACTION ==================

def get_segment_boundaries(total_duration, segment_duration=60.0):
    """
    Calculate time boundaries for 3 segments matching audio extraction strategy.
    
    Args:
        total_duration: Total duration of piece in seconds
        segment_duration: Target duration for each segment (default: 60s)
    
    Returns:
        list of tuples: [(start_begin, start_end), (mid_begin, mid_end), (end_begin, end_end)]
    """
    if total_duration < segment_duration:
        return None
    
    # Case 1: Long pieces (>= 180s) - extract 3 distinct 60s segments
    if total_duration >= 180:
        # Start: 0-60s
        start_segment = (0.0, segment_duration)
        
        # Middle: centered 60s
        mid_point = total_duration / 2
        middle_segment = (mid_point - segment_duration/2, mid_point + segment_duration/2)
        
        # End: last 60s
        end_segment = (total_duration - segment_duration, total_duration)
        
    # Case 2: Short pieces (60-180s) - split proportionally into 3 equal segments
    else:
        segment_length = total_duration / 3
        start_segment = (0.0, segment_length)
        middle_segment = (segment_length, 2 * segment_length)
        end_segment = (2 * segment_length, total_duration)
    
    return [start_segment, middle_segment, end_segment]


def filter_notes_by_time_range(all_notes, control_changes, time_start, time_end):
    """
    Filter notes and control changes to a specific time range.
    Shifts all timestamps so the segment starts at time 0.
    
    Args:
        all_notes: List of (pitch, start, end, velocity, duration) tuples
        control_changes: List of (time, value) tuples
        time_start: Start time of segment
        time_end: End time of segment
    
    Returns:
        tuple: (filtered_notes, filtered_control_changes)
    """
    filtered_notes = []
    for pitch, start, end, velocity, dur in all_notes:
        # Include notes that overlap with the segment
        if start < time_end and end > time_start:
            # Clip note boundaries to segment
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


def extract_notes_from_midi(midi_path):
    """
    Extract note events from a MIDI file.
    Returns list of (note_pitch, note_start_time, note_end_time, velocity, duration).
    """
    notes = []
    
    try:
        midi_file = mido.MidiFile(midi_path)
    except Exception as e:
        # Return None for corrupted or invalid MIDI files
        return None
    
    # Calculate tempo (default to 500000 microseconds per beat = 120 BPM)
    tempo = 500000  # microseconds per beat
    ticks_per_beat = midi_file.ticks_per_beat
    
    # Process all tracks, accumulate time
    current_time = 0.0
    active_notes = {}  # track_idx -> {pitch -> start_time}
    all_notes = []
    control_changes = []
    
    for track_idx, track in enumerate(midi_file.tracks):
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
                    
            elif msg.type == 'control_change' and msg.control == 64:  # Sustain pedal
                control_changes.append((current_time, msg.value))
    
    if not all_notes:
        return None
    
    return all_notes, control_changes


def compute_midi_features_from_notes(all_notes, control_changes, duration):
    """
    Extract MIDI features from a list of notes and control changes.
    
    Args:
        all_notes: List of (pitch, start, end, velocity, duration) tuples
        control_changes: List of (time, value) tuples for sustain pedal
        duration: Duration of the segment in seconds
    
    Returns:
        numpy array of shape (29,) with features or None if no notes
    """
    if not all_notes:
        return None

    # ===== PC Histogram (12) =====
    pc_histogram = np.zeros(12, dtype=np.float32)
    note_durations = []
    note_pitches = []
    
    for pitch, start, end, velocity, dur in all_notes:
        pitch_class = pitch % 12
        pc_histogram[pitch_class] += 1
        note_durations.append(dur)
        note_pitches.append(pitch)
    
    # Normalize PC histogram
    pc_histogram = pc_histogram / (len(all_notes) + 1e-8)
    
    # Convert to numpy arrays for faster computation
    note_durations = np.array(note_durations, dtype=np.float32)
    note_pitches = np.array(note_pitches, dtype=np.float32)
    

    # ===== Note Stats (6) =====
    mean_duration = np.mean(note_durations) if len(note_durations) > 0 else 0.0
    std_duration = np.std(note_durations) if len(note_durations) > 0 else 0.0
    
    # Categorize note types based on duration
    staccato_count = np.sum(note_durations < 0.1)
    sustained_count = np.sum(note_durations > 0.5)
    
    staccato_ratio = staccato_count / (len(all_notes) + 1e-8)
    sustained_ratio = sustained_count / (len(all_notes) + 1e-8)
    
    # Articulation: proportion of distinct note attacks
    articulation_ratio = min(len(all_notes) / (duration + 1e-8) / 10.0, 1.0)
    
    # Legato: check overlapping notes
    sorted_notes = sorted(all_notes, key=lambda n: n[1])
    legato_count = 0
    for i in range(1, len(sorted_notes)):
        if sorted_notes[i][1] < sorted_notes[i-1][2]:
            legato_count += 1
    legato_ratio = legato_count / max(len(all_notes) - 1, 1)
    
    note_stats = np.array([
        float(mean_duration),
        float(std_duration),
        float(staccato_ratio),
        float(sustained_ratio),
        float(articulation_ratio),
        float(legato_ratio)
    ], dtype=np.float32)
    

    # ===== Register Usage (5) =====
    mean_pitch = np.mean(note_pitches)
    std_pitch = np.std(note_pitches)
    
    # Pitch ranges for 88-key piano (A0=21 to C8=108)
    bass_fraction = np.sum(note_pitches < 54) / (len(note_pitches) + 1e-8)
    middle_fraction = np.sum((note_pitches >= 54) & (note_pitches < 72)) / (len(note_pitches) + 1e-8)
    treble_fraction = np.sum(note_pitches >= 72) / (len(note_pitches) + 1e-8)
    
    register_usage = np.array([
        float(mean_pitch),
        float(std_pitch),
        float(bass_fraction),
        float(middle_fraction),
        float(treble_fraction)
    ], dtype=np.float32)
    

    # ===== Note Density (3) =====
    mean_notes_per_second = len(all_notes) / (duration + 1e-8)
    
    # Mean simultaneous notes: count overlapping notes at each time point
    time_points = []
    for pitch, start, end, velocity, dur in all_notes:
        time_points.append((start, 1))    # note start
        time_points.append((end, -1))     # note end
    
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
        float(max_simultaneous)
    ], dtype=np.float32)
    

    # ===== Pedal Usage (3) =====
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
        float(pedal_variance)
    ], dtype=np.float32)
    

    # ===== Combine all BASIC features =====
    basic_features = np.concatenate([
        pc_histogram,      # 12
        note_stats,        # 6
        register_usage,    # 5
        note_density,      # 3
        pedal_usage        # 3
    ]).astype(np.float32)
    
    # ===== ADVANCED HARMONIC & DYNAMIC FEATURES =====
    harmonic_features = compute_pc_transition_matrix(all_notes)
    key_features = compute_key_features(all_notes, pc_histogram)
    intervallic_features = compute_intervallic_profile(all_notes)
    dynamic_features = compute_dynamic_features(all_notes, control_changes, duration)
    
    # Combine all features
    full_features = np.concatenate([
        basic_features,         # 29
        harmonic_features,      # 8
        key_features,           # 5
        intervallic_features,   # 6
        dynamic_features        # 16
    ]).astype(np.float32)
    
    return full_features


def compute_midi_features_3_segments(midi_path):
    """
    Extract MIDI features from 3 temporal segments of a MIDI file.
    
    Returns:
        tuple: (features_start, features_middle, features_end) or None if extraction fails
        Each features array has shape (29,)
    """
    result = extract_notes_from_midi(midi_path)
    
    if result is None:
        return None
    
    all_notes, control_changes = result
    
    if not all_notes:
        return None
    
    # Calculate total duration
    max_time = max(note[2] for note in all_notes)
    total_duration = max_time if max_time > 0 else 1.0
    
    # Get segment boundaries
    segments = get_segment_boundaries(total_duration)
    
    if segments is None:
        return None
    
    # Extract features for each segment
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
            # If one segment has no notes, return zeros
            features = np.zeros(29, dtype=np.float32)
        
        features_list.append(features)
    
    return tuple(features_list)


def get_midi_path_from_audio_path(audio_filename, metadata_df):
    """
    Given an audio filename, find the corresponding MIDI filename from the metadata CSV.

    Args:
        audio_filename (str): Audio filename to look up.
        metadata_df (pd.DataFrame): DataFrame containing metadata with 'audio_filename' and 'midi_filename' columns.
    """
    matches = metadata_df[metadata_df['audio_filename'] == audio_filename]
    if len(matches) > 0:
        return matches.iloc[0]['midi_filename']
    return None


# ================== MAIN ==================

print(f"Loading metadata from: {METADATA_PATH}")
metadata = pd.read_csv(METADATA_PATH)
print(f"Loaded {len(metadata)} files from CSV")

# Load existing labels to match audio and MIDI files
print(f"Loading labels from: {LABELS_PATH}")
labels_df = pd.read_csv(LABELS_PATH)
print(f"Loaded {len(labels_df)} labels")

print(f"Processing MIDI files...")

pbar = tqdm(total=len(labels_df), desc="Processing")

for idx, row in labels_df.iterrows():
    pbar.update(1)
    
    audio_filename = row['audio_filename']
    file_id = f"{int(row['id']):04d}"  # Format ID with leading zeros
    
    # Find MIDI filename from metadata
    midi_filename = get_midi_path_from_audio_path(audio_filename, metadata)
    
    if midi_filename is None:
        continue
    
    midi_path = DATASET_DIR / midi_filename
    
    # Skip if file doesn't exist
    if not midi_path.exists():
        continue
    
    try:
        features_tuple = compute_midi_features_3_segments(str(midi_path))
        
        if features_tuple is None:
            continue
        
        features_start, features_middle, features_end = features_tuple
        
        # Save features with same ID as audio features and segment suffix
        np.save(OUTPUT_DIR / f"{file_id}_midi_start.npy", features_start)
        np.save(OUTPUT_DIR / f"{file_id}_midi_middle.npy", features_middle)
        np.save(OUTPUT_DIR / f"{file_id}_midi_end.npy", features_end)
        
    except KeyboardInterrupt:
        raise  # Re-raise keyboard interrupt to allow clean exit
    except Exception as e:
        # Skip files that cause errors and continue processing
        continue

pbar.close()

print(f"Saved MIDI features to {OUTPUT_DIR}")
print("Done!")
