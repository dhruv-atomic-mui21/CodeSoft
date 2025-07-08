import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=512, fine_tune=True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, features_only=True)
        self.out_channels = self.backbone.feature_info[-1]['num_chs']
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.conv = nn.Conv2d(self.out_channels, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, x):
        feats = self.backbone(x)[-1]
        feats = self.bn(self.conv(feats))
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, num_layers=1, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        batch_size = features.size(0)
        seq_len = embeddings.size(1)
        avg_features = features.mean(dim=1)
        hidden_state = self.init_h(avg_features).unsqueeze(0)
        cell_state = self.init_c(avg_features).unsqueeze(0)
        outputs = []
        alphas = []
        for t in range(seq_len):
            attention_weighted_encoding, alpha = self.attention(features, hidden_state.squeeze(0))
            alphas.append(alpha)
            lstm_input = torch.cat((embeddings[:, t, :], attention_weighted_encoding), dim=1).unsqueeze(1)
            output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
            output = self.linear(self.dropout(output.squeeze(1)))
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        alphas = torch.stack(alphas, dim=1)
        return outputs

class ImageCaptioningModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_size=512, hidden_size=1024, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = EncoderCNN(embed_size, fine_tune=True)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def training_step(self, batch, batch_idx):
        images, captions = batch
        if images is None:
            return None
        outputs = self(images, captions)
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        if images is None:
            return None
        with torch.no_grad():
            outputs = self(images, captions)
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            },
        }
