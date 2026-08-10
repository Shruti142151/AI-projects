import os
import numpy as np
import random

from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from music21 import stream
from tensorflow.keras.models import load_model


DATASET_FOLDER = "dataset"

def load_notes():

    notes = []

    midi_files = [
        file for file in os.listdir(DATASET_FOLDER)
        if file.endswith(".mid") or file.endswith(".midi")
    ]

    print(f"\nFound {len(midi_files)} MIDI files.\n")

    for file in midi_files:

        print(f"Reading: {file}")

        midi = converter.parse(os.path.join(DATASET_FOLDER, file))

        try:
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def prepare_sequences(notes):

    sequence_length = 50

    pitchnames = sorted(set(notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):

        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([note_to_int[n] for n in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    network_input = np.reshape(
        network_input,
        (n_patterns, sequence_length, 1)
    )

    network_input = network_input / float(len(pitchnames))

    network_output = to_categorical(network_output)

    return network_input, network_output, pitchnames


def build_model(network_input, n_vocab):

    model = Sequential()

    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(256))
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(Dropout(0.3))

    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def generate_music(model, network_input, pitchnames, n_vocab):

    start = random.randint(0, len(network_input) - 1)

    pattern = (network_input[start]*n_vocab).flatten().astype(int)

    int_to_note = {number: note for number, note in enumerate(pitchnames)}

    prediction_output = []

    for _ in range(200):

        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)

        result = int_to_note[index]

        prediction_output.append(result)

        pattern = np.append(pattern[1:], index)

    return prediction_output

def create_midi(prediction_output):

    from music21 import stream, note, chord

    offset = 0
    output_notes = []

    for pattern in prediction_output:

        if "." in pattern:
            notes_in_chord = pattern.split(".")
            notes = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.offset = offset
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="generated_music.mid")

    print("\nMusic generated successfully!")
    print("File saved as: generated_music.mid")


if __name__ == "__main__":


    MODEL_PATH = "music_model.h5"

    notes = load_notes()

    print(f"\nTotal Notes Collected: {len(notes)}")

    network_input, network_output, pitchnames = prepare_sequences(notes)

    n_vocab = len(pitchnames)

    if not os.path.exists(MODEL_PATH):

        print("\nNo saved model found. Training new model...")

        model = build_model(network_input, n_vocab)

        model.fit(
            network_input,
            network_output,
            epochs=3,
            batch_size=64
        )

        model.save(MODEL_PATH)

        print("\nModel trained and saved successfully!")

    else:

        print("\nLoading saved model...")

        model = load_model(MODEL_PATH)

        print("\nModel loaded successfully!")

    print("\nGenerating music...")

    prediction_output = generate_music(
        model,
        network_input,
        pitchnames,
        n_vocab
    )

    create_midi(prediction_output)