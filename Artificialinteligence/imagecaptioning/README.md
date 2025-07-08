# Image Captioning Project

This project implements an image captioning model using PyTorch Lightning. The model generates descriptive captions for images using an encoder-decoder architecture with attention.

## Project Structure

- `train.py`: Main training script to train the image captioning model.
- `config.yaml`: Configuration file for model and training parameters.
- `best-checkpoint.ckpt`: Saved best model checkpoint after training.
- `bpe.model`, `bpe.vocab`: SentencePiece tokenizer model and vocabulary files.
- `data/`: Dataset folder containing images and captions.
- `src/`: Source code modules including dataset, model, trainer, inference, and vocabulary utilities.
- `Imagecaptoningmodel.ipynb`: Main Jupyter notebook with the full pipeline (used as the base for this project).
- `ImageCaptioning_Colab_prototype.ipynb`: prototype file.

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Download the Flickr8k dataset**

You need to download the Flickr8k dataset from Kaggle. Follow these steps:

- Create a Kaggle account and generate an API token (`kaggle.json`).
- Place the `kaggle.json` file in your home directory under `.kaggle/` or configure it as per Kaggle API instructions.
- Run the following command to download and unzip the dataset:

```bash
kaggle datasets download -d adityajn105/flickr8k -p Artificialinteligence/imagecaptioning/data --unzip
```

3. **Prepare captions file**

Ensure the captions file `captions.txt` is present in the `data/` directory. The training script will extract captions for tokenizer training.

## Training

Run the training script from the `Artificialinteligence/imagecaptioning/` directory:

```bash
python train.py
```

This will:

- Train a SentencePiece BPE tokenizer if not already trained.
- Load and preprocess the dataset.
- Train the image captioning model with EfficientNet-B3 encoder and attention-based decoder.
- Save the best model checkpoint as `best-checkpoint.ckpt`.

## Inference

You can use the inference functions defined in `src/inference.py` to generate captions for new images using beam search or greedy decoding.

Example usage:

```python
from src.inference import load_model_for_inference, process_image_for_inference, generate_caption_beam_search
import sentencepiece as spm
from torchvision import transforms
import torch

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

# Load model
model = load_model_for_inference("best-checkpoint.ckpt", vocab_size=sp.get_piece_size())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare image
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])
image_tensor = process_image_for_inference("data/Images/example.jpg", transform, device)

# Generate caption
caption = generate_caption_beam_search(image_tensor, model, sp, beam_width=5, max_len=30, device=device)
print("Generated Caption:", caption)
```

## Notes

- The project uses PyTorch Lightning for training and checkpointing.
- The model uses an EfficientNet-B3 backbone for image feature extraction.
- The vocabulary is built using a SentencePiece BPE tokenizer.
- The dataset used is Flickr8k, a standard benchmark for image captioning.

## License

This project is for educational and research purposes.

---

If you have any questions or need assistance, please refer to the notebook `Imagecaptoningmodel.ipynb` for detailed explanations and code.
