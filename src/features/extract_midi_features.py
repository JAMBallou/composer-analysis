"""
extract_midi_features.py
-------------------------
Utility to extract features from the MAESTRO MIDI files and store them as indexed numpy arrays in ``results/features/midi/``.

MIDI feature file structure (xxx_midi.npy):
- Feature vector containing:
    - PC Histogram (12 values): Note count per pitch class (C through B)
    - Note Stats (6 values): Mean duration, std duration, staccato ratio, sustained ratio, articulation ratio, legato ratio
    - Register Usage (5 values): Mean pitch, std pitch, fraction bass notes, fraction middle notes, fraction treble notes
    - Note Density (3 values): Mean notes/second, mean simultaneous notes, max simultaneous notes
    - Pedal Usage (3 values): Average sustain pedal value, pedal-on fraction, pedal variance
    - Total feature vector length: 29
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import mido

# Compute paths relative to repo root (project dir)
REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = REPO_ROOT / "data" / "maestro" / "maestro-v3.0.0.csv"
DATASET_DIR = REPO_ROOT / "data" / "maestro" / "data"
OUTPUT_DIR = REPO_ROOT / "results" / "features" / "midi"
LABELS_PATH = REPO_ROOT / "results" / "features" / "labels.csv"

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


def compute_midi_features(midi_path):
    """
    Extract MIDI features from a MIDI file.
    
    Returns a numpy array of shape (29,) with the following features:
    - PC Histogram (12): Note counts per pitch class
    - Note Stats (6): Mean/std duration, staccato/sustained/articulation/legato ratios
    - Register Usage (5): Mean/std pitch, bass/middle/treble fractions
    - Note Density (3): Mean notes/s, mean simultaneous notes, max simultaneous notes
    - Pedal Usage (3): Avg sustain value, pedal-on fraction, pedal variance
    """
    result = extract_notes_from_midi(midi_path)
    
    if result is None:
        return None
    
    all_notes, control_changes = result
    
    if not all_notes:
        return None
    
    # Calculate total duration
    max_time = max(note[2] for note in all_notes)
    duration = max_time if max_time > 0 else 1.0
    

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
    
    
    # ===== Combine all features =====
    features = np.concatenate([
        pc_histogram,      # 12
        note_stats,        # 6
        register_usage,    # 5
        note_density,      # 3
        pedal_usage        # 3
    ]).astype(np.float32)
    
    return features



def get_midi_path_from_audio_path(audio_filename, metadata_df):
    """
    Given an audio filename, find the corresponding MIDI filename from the metadata CSV.
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
        features = compute_midi_features(str(midi_path))
        
        if features is None:
            continue
        
        # Save features with same ID as audio features
        np.save(OUTPUT_DIR / f"{file_id}_midi.npy", features)
        
    except KeyboardInterrupt:
        raise  # Re-raise keyboard interrupt to allow clean exit
    except Exception as e:
        # Skip files that cause errors and continue processing
        continue

pbar.close()

print(f"Saved MIDI features to {OUTPUT_DIR}")
print("Done!")
