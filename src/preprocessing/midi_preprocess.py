"""
midi_preprocess.py
------------------
Extracts note, tempo, and key information from MIDI files.
"""

import pretty_midi
import numpy as np

def extract_midi_features(midi_path: str):
    """
    Returns structured MIDI features (e.g., pitch, duration, velocity histograms).
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append([note.pitch, note.start, note.end, note.velocity])
    notes = np.array(notes)
    # TODO: compute histogram features, average velocity, tempo curve
    return notes
