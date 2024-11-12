from flask import Flask, request, jsonify, send_file
import numpy as np
import pretty_midi
import collections
import pandas as pd
import torch
import torch.nn.functional as F
import os
import tempfile
from typing import Tuple

app = Flask(__name__)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Load the PyTorch model
model = torch.load("trained_structured_v3.1.1_epoch26.pth")
model.eval()  # Set model to evaluation mode

# Sampling rate for audio playback
_SAMPLING_RATE = 16000
key_order = ['pitch', 'step', 'duration']

@app.route('/')
def index():
    return app.send_static_file('../frontend/index.html')

# Function to process MIDI file and extract features
def process_midi_file(file_path):
    pm = pretty_midi.PrettyMIDI(file_path)
    return midi_to_notes(pm)

def midi_to_notes(pm: pretty_midi.PrettyMIDI) -> pd.DataFrame:
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def predict_next_note(
    notes: np.ndarray,
    model: torch.nn.Module,
    temperature: float = 1.0
) -> Tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""
    assert temperature > 0

    # Convert inputs to PyTorch tensor and add batch dimension
    inputs = torch.tensor(notes, dtype=torch.float32).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
        pitch_logits, step, duration = outputs['pitch'], outputs['step'], outputs['duration']

        # Apply temperature to pitch logits
        pitch_logits /= temperature
        pitch = torch.multinomial(F.softmax(pitch_logits, dim=-1), num_samples=1)
        pitch = pitch.item()
        step = max(0, step.item())
        duration = max(0, duration.item())

    return int(pitch), float(step), float(duration)

def create_sequences(
    dataset: torch.utils.data.Dataset,
    seq_length: int,
    vocab_size=128,
) -> torch.utils.data.Dataset:
    """Returns PyTorch Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    sequences = []
    for i in range(len(dataset) - seq_length):
        sequence = dataset[i:i + seq_length]
        inputs = sequence[:-1]
        labels_dense = sequence[-1]
        labels = {key: labels_dense[idx] for idx, key in enumerate(key_order)}
        inputs = inputs / torch.tensor([vocab_size, 1.0, 1.0])  # Normalize
        sequences.append((inputs, labels))
    
    return sequences

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file to a temporary location
    _, temp_file_path = tempfile.mkstemp(suffix='.mid')
    file.save(temp_file_path)

    # Process the MIDI file
    input_data = process_midi_file(temp_file_path)
    
    # Prepare input notes for prediction
    train_notes = np.stack([input_data[key] for key in key_order], axis=1)
    notes_ds = torch.tensor(train_notes, dtype=torch.float32)

    temperature = 2.0
    num_predictions = 120
    
    sample_notes = np.stack([input_data[key] for key in key_order], axis=1)
    seq_length = 25
    vocab_size = 128

    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
    input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))
    
    # Convert generated notes to a MIDI file
    _, predicted_file_path = tempfile.mkstemp(suffix='.mid')
    notes_to_midi(generated_notes, out_file=predicted_file_path)

    # Delete the temporary input MIDI file
    os.remove(temp_file_path)

    # Return the predicted MIDI file path in the response
    return send_file(predicted_file_path, as_attachment=True)

def notes_to_midi(notes_df, out_file='output.mid', instrument_name='Acoustic Grand Piano'):
    out_pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    for index, row in notes_df.iterrows():
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(row['pitch']),
            start=row['start'],
            end=row['end'])
        instrument.notes.append(note)
    out_pm.instruments.append(instrument)
    out_pm.write(out_file)
    return out_pm

if __name__ == '__main__':
    app.run(port=8000, debug=True)
