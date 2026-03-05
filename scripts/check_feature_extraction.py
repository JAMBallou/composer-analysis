"""
check_feature_extraction.py
===========================
Diagnose which files were skipped during feature extraction and why.
"""

import pandas as pd
from pathlib import Path

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
features_dir = repo_root / "outputs" / "features"
labels_csv = features_dir / "labels.csv"
audio_dir = features_dir / "audio"
midi_dir = features_dir / "midi"
dataset_dir = repo_root / "data" / "maestro" / "data"
metadata_csv = repo_root / "data" / "maestro" / "maestro-v3.0.0.csv"

# Load data
print("Loading data...")
labels_df = pd.read_csv(labels_csv)
metadata_df = pd.read_csv(metadata_csv)

print(f"\nTotal samples in labels.csv: {len(labels_df)}")

# Filter to Trial 2 composers
composers = ["Frédéric Chopin", "Franz Schubert"]
trial2_df = labels_df[labels_df["composer"].isin(composers)]
print(f"Trial 2 samples (Chopin + Schubert): {len(trial2_df)}")

# Check which have complete feature files
print("\n" + "="*80)
print("CHECKING FEATURE FILES FOR TRIAL 2")
print("="*80)

missing_audio = []
missing_midi = []
complete_files = []

for idx, row in trial2_df.iterrows():
    file_id = str(row["id"])
    
    # Check audio files
    audio_files = [
        audio_dir / f"{file_id}_mel_start.npy",
        audio_dir / f"{file_id}_mel_middle.npy",
        audio_dir / f"{file_id}_mel_end.npy",
        audio_dir / f"{file_id}_aux_start.npy",
        audio_dir / f"{file_id}_aux_middle.npy",
        audio_dir / f"{file_id}_aux_end.npy",
    ]
    
    # Check MIDI files
    midi_files = [
        midi_dir / f"{file_id}_midi_start.npy",
        midi_dir / f"{file_id}_midi_middle.npy",
        midi_dir / f"{file_id}_midi_end.npy",
    ]
    
    has_all_audio = all(f.exists() for f in audio_files)
    has_all_midi = all(f.exists() for f in midi_files)
    
    if not has_all_audio:
        missing_audio.append((file_id, row["composer"], row["audio_filename"]))
    
    if not has_all_midi:
        missing_midi.append((file_id, row["composer"], row["audio_filename"]))
    
    if has_all_audio and has_all_midi:
        complete_files.append(file_id)

print(f"\nComplete (have both audio + MIDI): {len(complete_files)}")
print(f"Missing audio features: {len(missing_audio)}")
print(f"Missing MIDI features: {len(missing_midi)}")

# Show which files are missing and why
if missing_audio:
    print("\n" + "="*80)
    print("FILES MISSING AUDIO FEATURES")
    print("="*80)
    for file_id, composer, audio_file in missing_audio[:10]:
        audio_path = dataset_dir / audio_file
        exists = audio_path.exists()
        print(f"\nID: {file_id} ({composer})")
        print(f"  File: {audio_file}")
        print(f"  On disk: {exists}")
        if exists:
            try:
                import librosa
                y, sr = librosa.load(str(audio_path), sr=22050)
                duration = len(y) / sr
                print(f"  Duration: {duration:.1f}s")
                if duration < 60:
                    print(f"  -> SKIPPED: Too short (< 60s minimum)")
            except Exception as e:
                print(f"  -> ERROR: {str(e)[:80]}")

if missing_midi:
    print("\n" + "="*80)
    print("FILES MISSING MIDI FEATURES")
    print("="*80)
    for file_id, composer, audio_file in missing_midi[:10]:
        # Find MIDI filename
        matches = metadata_df[metadata_df["audio_filename"] == audio_file]
        if len(matches) > 0:
            midi_file = matches.iloc[0]["midi_filename"]
            midi_path = dataset_dir / midi_file
            exists = midi_path.exists()
            print(f"\nID: {file_id} ({composer})")
            print(f"  Audio: {audio_file}")
            print(f"  MIDI: {midi_file}")
            print(f"  On disk: {exists}")
            if exists:
                try:
                    import mido
                    midi_file_obj = mido.MidiFile(str(midi_path))
                    # Count notes
                    note_count = 0
                    for track in midi_file_obj.tracks:
                        for msg in track:
                            if msg.type == "note_on" and msg.velocity > 0:
                                note_count += 1
                    print(f"  Notes: {note_count}")
                    if note_count == 0:
                        print(f"  -> SKIPPED: No notes (invalid/empty MIDI)")
                except Exception as e:
                    print(f"  -> ERROR: {str(e)[:80]}")
        else:
            print(f"\nID: {file_id} ({composer})")
            print(f"  Audio: {audio_file}")
            print(f"  -> ERROR: MIDI filename not in metadata")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Chopin samples with complete features: {len([x for x in complete_files if any(row['id'] == x and row['composer'] == 'Frédéric Chopin' for _, row in trial2_df.iterrows())])}")
print(f"Schubert samples with complete features: {len([x for x in complete_files if any(row['id'] == x and row['composer'] == 'Franz Schubert' for _, row in trial2_df.iterrows())])}")
print(f"Total usable for Trial 2: {len(complete_files)}")
print(f"Expected iterations/epoch (batch=16): {len(complete_files) / 16:.1f}")
