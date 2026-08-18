import torch
import torch.nn as nn
from torch.utils.data import Dataset
import re
from collections import Counter
import json
import os

class SimpleTokenizer:
    def __init__(self, max_vocab_size=20000, max_length=200):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit_on_texts(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.clean_text(text)
            word_counts.update(tokens)
            
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def text_to_sequence(self, text):
        tokens = self.clean_text(text)
        seq = [self.word2idx.get(w, 1) for w in tokens]
        if len(seq) < self.max_length:
            seq = seq + [0] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        return seq

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                "max_vocab_size": self.max_vocab_size,
                "max_length": self.max_length,
                "word2idx": self.word2idx
            }, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        tok = cls(max_vocab_size=data["max_vocab_size"], max_length=data["max_length"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        return tok


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.labels = labels
        self.sequences = [tokenizer.text_to_sequence(t) for t in texts]
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class ScamBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super(ScamBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits.squeeze(-1)
