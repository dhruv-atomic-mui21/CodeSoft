import json
import nltk
from collections import Counter
import os
import sentencepiece as spm

nltk.download("punkt")

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return self.idx

def build_vocab(captions_file, freq_threshold=5):
    # Check if captions file exists
    if not os.path.exists(captions_file):
        print(f"Error: Captions file not found at {captions_file}")
        return None  # Return None or raise an error

    with open(captions_file, "r") as f:
        data = json.load(f)

    counter = Counter()
    # Assuming data is a dictionary where keys are image IDs and values are lists of captions
    for cap_list in data.values():
        for cap in cap_list:
            tokens = nltk.word_tokenize(cap.lower())
            counter.update(tokens)

    vocab = Vocabulary()
    for token in ["<pad>", "<start>", "<end>", "<unk>"]:
        vocab.add_word(token)
    for word, freq in counter.items():
        if freq >= freq_threshold:
            vocab.add_word(word)

    return vocab

def train_sentencepiece_tokenizer(input_file: str, model_prefix: str, vocab_size: int = 4000):
    """
    Train a SentencePiece BPE tokenizer on the given input file.

    Args:
        input_file (str): Path to the text file containing training data.
        model_prefix (str): Prefix for the output model files.
        vocab_size (int): Vocabulary size for the tokenizer.
    """
    spm.SentencePieceTrainer.Train(
        f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    )
