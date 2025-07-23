from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np

# Define the model class (same as in notebook)
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
import random

class HandwritingRNN(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 256, num_layers: int = 2, output_size: int = 3, dropout_prob: float = 0.2) -> None:
        super(HandwritingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_prob
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        out, hidden = self.lstm(x, hidden)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HandwritingRNN()
model.load_state_dict(torch.load('handwriting_rnn.pth', map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)

# Load real stroke data for seed generation
stroke_data = np.load('strokes.npy', allow_pickle=True)

def sample_real_stroke_seed() -> list:
    """
    Sample a random stroke sequence from the real stroke data to use as seed.
    """
    seq = random.choice(stroke_data)
    # Convert to list of [x, y, pen] and ensure pen is 0 or 1
    seed = []
    for point in seq:
        x, y, pen = point
        pen_state = 1 if pen > 0 else 0
        seed.append([x, y, pen_state])
    return seed

def generate_sequence(model: HandwritingRNN, seed_seq: List[List[float]], length: int = 300) -> List[List[float]]:
    model.eval()
    generated = []
    input_seq = torch.tensor(seed_seq, dtype=torch.float32).unsqueeze(0).to(device)
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            out, hidden = model(input_seq, hidden)
            next_point = out[:, -1, :].cpu().numpy()
            generated.append(next_point[0])
            input_seq = out[:, -1:, :]
    return np.array(generated).tolist()

@app.route('/')
def index():
    return render_template('index.html')

def offset_strokes(strokes: list, x_offset: float) -> list:
    """
    Offset the x coordinate of all strokes by x_offset.
    """
    return [[x + x_offset, y, pen] for x, y, pen in strokes]

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', None)
    if text is None or not isinstance(text, str) or text.strip() == '':
        return jsonify({'error': 'Text input is required'}), 400
    try:
        all_strokes = []
        x_offset = 0.0
        for char in text.strip():
            seed = sample_real_stroke_seed()
            generated_strokes = generate_sequence(model, seed, length=100)
            # Offset strokes to avoid overlap
            offsetted = offset_strokes(generated_strokes, x_offset)
            all_strokes.extend(offsetted)
            x_offset += 15.0  # adjust spacing between characters
        return jsonify({'generated_strokes': all_strokes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_new', methods=['GET'])
def generate_new():
    try:
        # Default seed: a simple stroke to start generation
        seed = [[0.0, 0.0, 1], [1.0, 0.0, 0], [1.0, 1.0, 0], [0.0, 1.0, 0], [0.0, 0.0, 0]]
        generated_strokes = generate_sequence(model, seed, length=500)
        return jsonify({'generated_strokes': generated_strokes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
