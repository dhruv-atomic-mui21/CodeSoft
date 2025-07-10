import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
from torchvision import transforms
from PIL import Image
import sentencepiece as spm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from inference import load_model_for_inference, process_image_for_inference, generate_caption_beam_search
import sentencepiece as spm
import torch
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory

app = Flask(__name__, template_folder='frontend/templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model once at startup
sp = spm.SentencePieceProcessor()
sp_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe.model")
sp.load(sp_model_path)

ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best-checkpoint.ckpt")
model = load_model_for_inference(ckpt_path, vocab_size=sp.get_piece_size())
model.to(device)
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    error = None
    image_url = None
    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No image part in the request."
            return render_template('index.html', caption=caption, error=error)
        file = request.files['image']
        if file.filename == '':
            error = "No selected file."
            return render_template('index.html', caption=caption, error=error)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_tensor = process_image_for_inference(filepath, inference_transform, device)
            if image_tensor is not None:
                caption = generate_caption_beam_search(image_tensor, model, sp, beam_width=5, max_len=30, device=device)
                image_url = url_for('uploaded_file', filename=file.filename)
            else:
                error = "Failed to process the image."
    return render_template('index.html', caption=caption, error=error, image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/caption', methods=['POST'])
def api_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    image_tensor = process_image_for_inference(filepath, inference_transform, device)
    if image_tensor is None:
        return jsonify({'error': 'Failed to process the image.'}), 500
    caption = generate_caption_beam_search(image_tensor, model, sp, beam_width=5, max_len=30, device=device)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
