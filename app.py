from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import pretty_midi
import collections
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import os
import tempfile
import tensorflow.keras.backend as K

app = Flask(__name__)

model = keras.models.load_model("mooot.h5")

@app.route('/')
def index():
    return render_template('form.html')

# Define function to process MIDI file and extract features
def process_midi_file(file_path):
    pm = pretty_midi.PrettyMIDI(file_path)
    return midi_to_notes(pm)

# @keras.saving.register_keras_serializable(package="my_package", name="mse_with_positive_pressure")
# def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
#   mse = (y_true - y_pred) ** 2
#   positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
#   return tf.reduce_mean(mse + positive_pressure)

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
    model: keras.Model,
    temperature: float = 1.0) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""
    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print("1")
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    print("1")
    

    # Save the uploaded file to a temporary location
    _, temp_file_path = tempfile.mkstemp(suffix='.mid')
    file.save(temp_file_path)

    print("1")
    
    # Process the MIDI file
    input_data = process_midi_file(temp_file_path)

    print("1")
    
    # Generate MIDI notes using the model
    generated_notes = generate_notes(input_data, model)

    print("1")
    
    # Convert the generated notes to a MIDI file
    _, predicted_file_path = tempfile.mkstemp(suffix='.mid')
    notes_to_midi(generated_notes, out_file=predicted_file_path)
    print("1")
    # Delete the temporary input MIDI file
    os.remove(temp_file_path)
    print("1")
    # Return the predicted MIDI file path in the response
    return send_file(predicted_file_path, as_attachment=True)

def generate_notes(input_notes, model, temperature=2.0, num_predictions=120):
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

    return pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])

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
    app.run(debug=True)
