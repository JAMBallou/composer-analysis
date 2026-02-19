"""
advanced_midi_features.py
-------------------------
Advanced harmonic and dynamic MIDI features for composer classification.

Features extracted (64 total per segment):
- Original features (29)
- Harmonic Movement (8): PC transition matrix statistics
- Key Analysis (5): Key clarity, changes, tonal centroid
- Intervallic Profile (6): Melodic interval statistics
- Dynamics (16): Velocity-based dynamic features

Total: 64 features per segment
"""

import numpy as np


def compute_pc_transition_matrix(all_notes):
    """
    Compute 12x12 pitch class transition matrix and extract features.
    
    Args:
        all_notes: List of (pitch, start, end, velocity, duration) tuples
    
    Returns:
        numpy array of shape (8,) with harmonic movement features:
        - PC transition entropy (1)
        - Diagonal dominance (1)
        - Top 3 non-diagonal transitions (3)
        - Transition count (1) 
        - Mean transition distance (1)
        - Self-loop frequency (1)
    """
    if len(all_notes) < 2:
        return np.zeros(8, dtype=np.float32)
    
    # Sort by start time
    sorted_notes = sorted(all_notes, key=lambda x: x[1])
    
    # Build transition matrix
    transition_matrix = np.zeros((12, 12), dtype=np.float32)
    
    for i in range(len(sorted_notes) - 1):
        current_pc = sorted_notes[i][0] % 12
        next_pc = sorted_notes[i+1][0] % 12
        transition_matrix[current_pc, next_pc] += 1
    
    # Normalize
    totals = transition_matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    transition_prob = transition_matrix / totals
    
    # Feature 1: Shannon entropy of transitions
    # H = -sum(p * log(p))
    h_values = transition_prob[transition_prob > 0]
    entropy = -np.sum(h_values * np.log2(h_values + 1e-10))
    entropy = entropy / 12  # Normalize to [0, 1]
    
    # Feature 2: Diagonal dominance (self-loops)
    diagonal = np.diag(transition_prob)
    diagonal_dominance = np.mean(diagonal)
    
    # Feature 3-5: Top 3 non-diagonal transitions (sparse representation)
    non_diag_values = transition_prob.copy()
    np.fill_diagonal(non_diag_values, 0)
    top_3 = np.sort(non_diag_values.flatten())[-3:]
    top_3 = np.pad(top_3, (3 - len(top_3), 0), mode='constant')[::-1]
    
    # Feature 6: Total number of transitions
    transition_count = transition_matrix.sum()
    
    # Feature 7: Mean transition distance (chromatic distance between PCs)
    transition_distances = []
    for i in range(len(sorted_notes) - 1):
        pc1 = sorted_notes[i][0] % 12
        pc2 = sorted_notes[i+1][0] % 12
        dist = min(abs(pc2 - pc1), 12 - abs(pc2 - pc1))
        transition_distances.append(dist)
    
    mean_transition_dist = np.mean(transition_distances) if transition_distances else 0.0
    
    # Feature 8: Self-loop frequency (%)
    self_loops = np.diag(transition_matrix).sum()
    self_loop_freq = self_loops / (transition_count + 1e-8)
    
    harmonic_features = np.array([
        float(entropy),
        float(diagonal_dominance),
        float(top_3[0]),
        float(top_3[1]),
        float(top_3[2]),
        float(transition_count),
        float(mean_transition_dist),
        float(self_loop_freq)
    ], dtype=np.float32)
    
    return harmonic_features


def compute_key_features(all_notes, pc_histogram):
    """
    Compute key clarity, key changes, and tonal centroid features.
    Uses Krumhansl-Schmuckler algorithm with pitch class histogram.
    
    Args:
        all_notes: List of (pitch, start, end, velocity, duration) tuples
        pc_histogram: Normalized pitch class histogram (12,)
    
    Returns:
        numpy array of shape (5,) with key analysis features:
        - Key clarity score (0-1) (1)
        - Key change frequency (1)
        - Tonal centroid X (Tonnetz) (1)
        - Tonal centroid Y (Tonnetz) (1)
        - Tonal centroid variance (1)
    """
    # Krumhansl-Schmuckler profiles (major and minor keys)
    # Weights for each pitch class
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major_profile = major_profile / major_profile.sum()
    minor_profile = minor_profile / minor_profile.sum()
    
    # Correlate histogram with profiles
    pc_hist_norm = pc_histogram / (pc_histogram.sum() + 1e-8)
    
    # Check all 12 transpositions
    correlations_major = []
    correlations_minor = []
    
    for shift in range(12):
        major_shifted = np.roll(major_profile, shift)
        minor_shifted = np.roll(minor_profile, shift)
        
        # Pearson correlation
        corr_major = np.corrcoef(pc_hist_norm, major_shifted)[0, 1]
        corr_minor = np.corrcoef(pc_hist_norm, minor_shifted)[0, 1]
        
        correlations_major.append(corr_major if not np.isnan(corr_major) else 0.0)
        correlations_minor.append(corr_minor if not np.isnan(corr_minor) else 0.0)
    
    # Feature 1: Key clarity = max correlation (0-1 scale)
    key_clarity = max(max(correlations_major), max(correlations_minor))
    key_clarity = (key_clarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    # Feature 2: Key change frequency (from note movement)
    # Detect changes in dominant pitch class
    if len(all_notes) > 1:
        sorted_notes = sorted(all_notes, key=lambda x: x[1])
        # Split into 4 equal time windows
        window_size = len(sorted_notes) // 4
        if window_size > 0:
            pc_changes = 0
            prev_pc = sorted_notes[0][0] % 12
            
            for note in sorted_notes[1:]:
                curr_pc = note[0] % 12
                if curr_pc != prev_pc:
                    pc_changes += 1
                prev_pc = curr_pc
            
            key_change_freq = pc_changes / len(sorted_notes)
        else:
            key_change_freq = 0.0
    else:
        key_change_freq = 0.0
    
    # Features 3-4: Tonal centroid in Tonnetz space
    # Tonnetz coordinates: fifth (x) and major-third (y)
    # 12 pitches mapped to 2D Tonnetz space
    tonnetz_x = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]) / 12.0
    tonnetz_y = np.array([0, 3.5, 7, 3.5, 7, 3.5, 7, 3.5, 7, 3.5, 7, 3.5]) / 7.0
    
    # Weight by pitch class histogram
    tonal_centroid_x = np.sum(pc_histogram * tonnetz_x)
    tonal_centroid_y = np.sum(pc_histogram * tonnetz_y)
    
    # Feature 5: Tonal centroid variance (stability)
    tonal_variance = np.sqrt(tonal_centroid_x**2 + tonal_centroid_y**2)
    
    key_features = np.array([
        float(key_clarity),
        float(key_change_freq),
        float(tonal_centroid_x),
        float(tonal_centroid_y),
        float(tonal_variance)
    ], dtype=np.float32)
    
    return key_features


def compute_intervallic_profile(all_notes):
    """
    Compute melodic interval statistics and distribution.
    
    Args:
        all_notes: List of (pitch, start, end, voltage, duration) tuples
    
    Returns:
        numpy array of shape (6,) with interval features:
        - Mean melodic interval (semitones) (1)
        - Std of melodic interval (1)
        - Fraction of leaps (>5 semitones) (1)
        - Most common interval (semitones) (1)
        - Interval entropy (1)
        - Descending fraction (1)
    """
    if len(all_notes) < 2:
        return np.zeros(6, dtype=np.float32)
    
    # Sort by start time
    sorted_notes = sorted(all_notes, key=lambda x: x[1])
    
    # Compute intervals
    intervals = []
    for i in range(len(sorted_notes) - 1):
        pitch1 = sorted_notes[i][0]
        pitch2 = sorted_notes[i+1][0]
        interval = pitch2 - pitch1  # Signed interval in semitones
        intervals.append(interval)
    
    if not intervals:
        return np.zeros(6, dtype=np.float32)
    
    intervals = np.array(intervals, dtype=np.float32)
    
    # Feature 1-2: Mean and std of intervals
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    # Feature 3: Leap fraction (intervals > 5 semitones)
    leap_fraction = np.sum(np.abs(intervals) > 5) / len(intervals)
    
    # Feature 4: Most common interval (mode)
    rounded_intervals = np.round(intervals).astype(int)
    unique, counts = np.unique(rounded_intervals, return_counts=True)
    most_common_interval = unique[np.argmax(counts)]
    
    # Feature 5: Interval entropy (distribution diversity)
    unique_intervals, counts = np.unique(rounded_intervals, return_counts=True)
    interval_prob = counts / counts.sum()
    interval_entropy = -np.sum(interval_prob * np.log2(interval_prob + 1e-10))
    
    # Feature 6: Descending fraction
    descending_fraction = np.sum(intervals < 0) / len(intervals)
    
    intervallic_features = np.array([
        float(mean_interval),
        float(std_interval),
        float(leap_fraction),
        float(most_common_interval),
        float(interval_entropy),
        float(descending_fraction)
    ], dtype=np.float32)
    
    return intervallic_features


def compute_dynamic_features(all_notes, control_changes, duration):
    """
    Compute velocity-based dynamic features.
    
    Args:
        all_notes: List of (pitch, start, end, velocity, duration) tuples
        control_changes: List of (time, value) tuples for sustain pedal
        duration: Duration of the segment in seconds
    
    Returns:
        numpy array of shape (16,) with dynamic features:
        - Mean velocity (1)
        - Std velocity (1)
        - Min velocity (1)
        - Max velocity (1)
        - Dynamic range (max - min) (1)
        - Velocity CV (std/mean) (1)
        - Crescendo frequency (1)
        - Decrescendo frequency (1)
        - Mean crescendo slope (1)
        - Mean decrescendo slope (1)
        - Accent frequency (high velocity notes) (1)
        - Ppp frequency (very soft notes) (1)
        - Velocity changes per second (1)
        - Sudden dynamic shifts (>40 velocity) (1)
        - Velocity stability (low variability) (1)
        - Dynamic range ratio (range / max) (1)
    """
    if not all_notes:
        return np.zeros(16, dtype=np.float32)
    
    velocities = np.array([vel for pitch, start, end, vel, dur in all_notes], dtype=np.float32)
    
    # Features 1-6: Basic velocity statistics
    mean_velocity = np.mean(velocities)
    std_velocity = np.std(velocities)
    min_velocity = np.min(velocities)
    max_velocity = np.max(velocities)
    dynamic_range = max_velocity - min_velocity
    velocity_cv = std_velocity / (mean_velocity + 1e-8)
    
    # Features 7-10: Crescendo/decrescendo analysis
    sorted_notes = sorted(all_notes, key=lambda x: x[1])
    sorted_velocities = np.array([vel for pitch, start, end, vel, dur in sorted_notes], dtype=np.float32)
    sorted_times = np.array([start for pitch, start, end, vel, dur in sorted_notes], dtype=np.float32)
    
    velocity_deltas = np.diff(sorted_velocities)
    time_deltas = np.diff(sorted_times)
    
    # Avoid division by zero
    time_deltas[time_deltas == 0] = 1e-8
    velocity_slopes = velocity_deltas / time_deltas
    
    crescendos = velocity_slopes > 2  # Velocity increase > 2 units/sec
    decrescendos = velocity_slopes < -2  # Velocity decrease > 2 units/sec
    
    crescendo_freq = np.sum(crescendos) / len(velocity_slopes)
    decrescendo_freq = np.sum(decrescendos) / len(velocity_slopes)
    mean_crescendo_slope = np.mean(velocity_slopes[crescendos]) if np.any(crescendos) else 0.0
    mean_decrescendo_slope = np.mean(velocity_slopes[decrescendos]) if np.any(decrescendos) else 0.0
    
    # Features 11-12: Accent and ppp frequencies
    accent_threshold = np.percentile(velocities, 85)  # Top 15%
    ppp_threshold = np.percentile(velocities, 15)  # Bottom 15%
    
    accent_freq = np.sum(velocities > accent_threshold) / len(velocities)
    ppp_freq = np.sum(velocities < ppp_threshold) / len(velocities)
    
    # Feature 13: Velocity changes per second
    velocity_changes = np.sum(np.abs(velocity_deltas) > 0)
    velocity_changes_per_sec = velocity_changes / (duration + 1e-8)
    
    # Feature 14: Sudden dynamic shifts (>40 velocity units)
    sudden_shifts = np.sum(np.abs(velocity_deltas) > 40)
    
    # Feature 15: Dynamic stability (inverse of variability)
    dynamic_stability = 1.0 / (1.0 + velocity_cv)
    
    # Feature 16: Dynamic range ratio
    dynamic_range_ratio = dynamic_range / (max_velocity + 1e-8)
    
    dynamic_features = np.array([
        float(mean_velocity),
        float(std_velocity),
        float(min_velocity),
        float(max_velocity),
        float(dynamic_range),
        float(velocity_cv),
        float(crescendo_freq),
        float(decrescendo_freq),
        float(mean_crescendo_slope),
        float(mean_decrescendo_slope),
        float(accent_freq),
        float(ppp_freq),
        float(velocity_changes_per_sec),
        float(sudden_shifts),
        float(dynamic_stability),
        float(dynamic_range_ratio)
    ], dtype=np.float32)
    
    return dynamic_features
