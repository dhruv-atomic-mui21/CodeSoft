# Handwritten Text Generation

## Project Overview
A Flask web application that generates handwritten text strokes using a trained PyTorch Recurrent Neural Network (RNN) model.

## Features
- Generates handwriting sequences based on input text.
- Uses a pretrained HandwritingRNN model.
- Provides API endpoints for generating handwriting data.
- Serves a web interface for user interaction.

## Setup Instructions
1. Install Python and required packages (Flask, PyTorch, numpy).
2. Ensure the pretrained model file `handwriting_rnn.pth` and stroke data `strokes.npy` are present.
3. Run the Flask app:
   ```
   python app.py
   ```

## API Endpoints
- `GET /`: Serves the main web interface.
- `POST /generate`: Generates handwriting strokes for input text.
- `GET /generate_new`: Generates handwriting strokes with a default seed.

## Usage
- Access the web interface at `http://localhost:5000/`.
- Input text to generate corresponding handwritten strokes.
