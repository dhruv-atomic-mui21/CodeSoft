import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import sentencepiece as spm

from src.dataset import ImageCaptionDataset
from src.model import ImageCaptioningModel
from src.vocab import train_sentencepiece_tokenizer

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch.cuda.device_count())
    pl.seed_everything(seed)

def main():
    seed_everything()

    batch_size = 64
    max_epochs = 15
    learning_rate = 1e-4
    captions_file = "data/captions.txt"
    image_dir = "data/Images"
    bpe_model_prefix = "bpe"
    vocab_size = 4000

    captions_only_file = "captions_only.txt"
    with open(captions_file, 'r') as f_in, open(captions_only_file, 'w') as f_out:
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                parts = line.split(',')
                caption = ','.join(parts[1:]).strip()
                if caption.lower() != 'caption':
                    f_out.write(caption + '\n')
            elif '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2:
                    caption = parts[1].strip()
                    f_out.write(caption + '\n')

    if not os.path.exists(bpe_model_prefix + ".model"):
        print("Training SentencePiece tokenizer...")
        train_sentencepiece_tokenizer(captions_only_file, bpe_model_prefix, vocab_size)
    else:
        print("SentencePiece tokenizer model found, skipping training.")

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_prefix + ".model")

    all_samples = []
    with open(captions_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.lower().startswith("image"):
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    img_name = parts[0].strip()
                    caption = ','.join(parts[1:]).strip()
                    if img_name.lower() != 'image':
                        all_samples.append((img_name, caption))
            elif '\t' in line:
                parts = line.split('\t')
                if len(parts) == 2:
                    img_info, caption = parts
                    img_name = img_info.split('#')[0].strip()
                    all_samples.append((img_name, caption.strip()))

    print(f"Total samples: {len(all_samples)}")

    train_samples, val_samples = train_test_split(all_samples, test_size=0.10, random_state=42, shuffle=True)
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = ImageCaptionDataset(image_dir=image_dir, samples=train_samples, sp=sp, transform=train_transform)
    val_dataset = ImageCaptionDataset(image_dir=image_dir, samples=val_samples, sp=sp, transform=val_transform)

    num_workers = min(8, os.cpu_count() or 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=ImageCaptionDataset.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=ImageCaptionDataset.custom_collate_fn)

    model = ImageCaptioningModel(vocab_size=sp.get_piece_size(), embed_size=512, hidden_size=1024, learning_rate=learning_rate)

    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best-checkpoint')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
