import os
import random
import argparse
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau

nltk.download('punkt')

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        for token in ['<pad>', '<start>', '<end>', '<unk>']:
            self.add_word(token)
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])
    def __len__(self):
        return self.idx

def build_vocab(captions_file, freq_threshold=5):
    with open(captions_file, 'r') as f:
        data = f.readlines()
    counter = {}
    for line in data:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            caption = parts[1]
            tokens = nltk.word_tokenize(caption.lower())
            for token in tokens:
                counter[token] = counter.get(token, 0) + 1
    vocab = Vocabulary()
    for word, freq in counter.items():
        if freq >= freq_threshold:
            vocab.add_word(word)
    return vocab

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, vocab, samples=None, captions_file=None, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        if samples is not None:
            self.img_captions = samples
        elif captions_file is not None:
            self.captions_file = captions_file
            self.img_captions = self._load_data()
        else:
            raise ValueError("Either 'samples' or 'captions_file' must be provided.")
    def _load_data(self):
        img_captions = []
        with open(self.captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_info, caption = parts
                    img_name = img_info.split('#')[0]
                    img_captions.append((img_name, caption))
        return img_captions
    def __len__(self):
        return len(self.img_captions)
    def __getitem__(self, idx):
        img_name, caption = self.img_captions[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping sample.")
            return None, None
        if self.transform:
            image = self.transform(image)
        tokens = ['<start>'] + nltk.word_tokenize(caption.lower()) + ['<end>']
        indices = [self.vocab(token) for token in tokens]
        return image, torch.tensor(indices)
    @staticmethod
    def custom_collate_fn(batch):
        batch = [item for item in batch if item[0] is not None]
        if not batch:
            return None, None
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        max_len = max(len(cap) for cap in captions)
        padded_captions = torch.zeros((len(captions), max_len), dtype=torch.long)
        for i, cap in enumerate(captions):
            padded_captions[i, :len(cap)] = cap
        return images, padded_captions

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, fine_tune=True):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        if fine_tune:
            for param in list(resnet.layer4.parameters()):
                param.requires_grad = True
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(embed_size)
        self.relu = nn.ReLU()
    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = self.conv(features)
        features = self.bn(features)
        features = self.relu(features)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        batch_size = features.size(0)
        seq_len = embeddings.size(1)
        hidden_state = torch.zeros(1, batch_size, self.lstm.hidden_size).to(features.device)
        cell_state = torch.zeros(1, batch_size, self.lstm.hidden_size).to(features.device)
        outputs = []
        for t in range(seq_len):
            attention_weighted_encoding, alpha = self.attention(features, hidden_state[-1])
            lstm_input = torch.cat((embeddings[:, t, :], attention_weighted_encoding), dim=1).unsqueeze(1)
            output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
            output = self.linear(self.dropout(output.squeeze(1)))
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs

class ImageCaptioningModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_size=512, hidden_size=1024, learning_rate=1e-4):
        super().__init__()
        self.encoder = EncoderCNN(embed_size, fine_tune=True)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.learning_rate = learning_rate
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    def training_step(self, batch, batch_idx):
        images, captions = batch
        outputs = self(images, captions)
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        images, captions = batch
        with torch.no_grad():
            outputs = self(images, captions)
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            },
            'gradient_clip_val': 1.0
        }

# Main function to run training with CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument('--image_dir', type=str, default='data/Flicker8k_Dataset', help='Directory with images')
    parser.add_argument('--captions_file', type=str, default='data/Flickr8k_text/Flickr8k.token.txt', help='Captions file path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=15, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--freq_threshold', type=int, default=5, help='Frequency threshold for vocabulary')
    args = parser.parse_args()

    print("Building vocabulary...")
    vocab = build_vocab(args.captions_file, freq_threshold=args.freq_threshold)
    print(f"Vocabulary size: {len(vocab)}")

    print("Loading dataset...")
    full_dataset = ImageCaptionDataset(args.image_dir, vocab, captions_file=args.captions_file, transform=None)

    print("Splitting dataset...")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    num_workers = min(8, os.cpu_count() or 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=ImageCaptionDataset.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=ImageCaptionDataset.custom_collate_fn)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}, Num workers: {num_workers}")

    model = ImageCaptioningModel(vocab_size=len(vocab), embed_size=512, hidden_size=1024, learning_rate=args.learning_rate)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best-checkpoint')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.max_epochs,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n--- Training Metrics ---")
    print(f"Best validation loss: {trainer.callback_metrics.get('val_loss').item():.4f}")
    print(f"Epochs trained: {trainer.current_epoch}")
    print(f"Training time: See TensorBoard logs or callback logs")

    print("\n--- Execution Data ---")
    print(f"Device: {trainer.strategy.root_device}")
    print(f"Number of GPUs used: {trainer.num_devices}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of workers for DataLoaders: {num_workers}")
    print(f"Model saved at: best-checkpoint.ckpt")

if __name__ == "__main__":
    main()
