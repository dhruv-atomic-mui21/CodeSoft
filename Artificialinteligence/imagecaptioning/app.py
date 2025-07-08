import os
from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import sentencepiece as spm

from src.inference import load_model_for_inference, process_image_for_inference, generate_caption_beam_search

app = Flask(__name__)
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
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_tensor = process_image_for_inference(filepath, inference_transform, device)
            if image_tensor is not None:
                caption = generate_caption_beam_search(image_tensor, model, sp, beam_width=5, max_len=30, device=device)
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
