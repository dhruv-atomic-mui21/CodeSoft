from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import torch

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, samples, sp, transform=None):
        """
        image_dir: directory containing images
        samples: list of (image_name, caption) tuples
        sp: SentencePieceProcessor tokenizer
        transform: torchvision transforms to apply to images
        """
        self.image_dir = image_dir
        self.samples = samples
        self.sp = sp
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Encode caption using SentencePiece tokenizer with BOS and EOS tokens
        encoded = [self.sp.bos_id()] + self.sp.encode(caption, out_type=int) + [self.sp.eos_id()]
        return image, torch.tensor(encoded)

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
