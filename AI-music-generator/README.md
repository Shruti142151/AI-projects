AI Music Generator

Overview

This project is an AI-based Music Generator developed using Python, TensorFlow, Keras, and Music21. It learns musical patterns from MIDI files and generates new melodies using a trained neural network.

Features

- Reads MIDI files for training
- Uses an LSTM neural network for sequence learning
- Generates new musical note sequences
- Saves the generated music as a MIDI file
- Beginner-friendly implementation

Technologies Used

- Python
- TensorFlow
- Keras
- Music21
- NumPy

Project Structure

- "music_generator.py" – Main application
- "dataset/" – Training MIDI files
- "generated_music.mid" – Generated music output
- "music_model.h5" – Trained model

How to Run

1. Install the required libraries.
2. Place MIDI files inside the "dataset" folder.
3. Run:
   python music_generator.py
4. The generated music will be saved as "generated_music.mid".

Future Improvements

- Train on larger datasets
- Generate longer compositions
- Add multiple music styles
- Build a graphical user interface
