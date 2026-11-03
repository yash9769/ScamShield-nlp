# =============================================================================
# FILE: src/models/bilstm.py          (BUILDING BLOCKS of the BiLSTM model)
# PURPOSE: Defines three classes used by training (train_lstm.py), by the
#          comparison script and by the live predictor:
#            1. SimpleTokenizer - turns text into lists of integers
#            2. TextDataset     - wraps those integer lists so PyTorch can batch them
#            3. ScamBiLSTM      - the neural network itself
# No training happens in this file; it only DEFINES things.
# =============================================================================
import torch
import torch.nn as nn                       # neural-network layers (Embedding, LSTM, Linear, ...)
from torch.utils.data import Dataset        # base class for PyTorch datasets
import re
from collections import Counter             # dictionary that counts occurrences
import json
import os

class SimpleTokenizer:
    """A hand-written word-level tokenizer (no library).

    A neural network only understands NUMBERS, so every word is mapped to an
    integer id. Example: "win free prize" -> [532, 87, 1204].
    Reserved ids: 0 = <pad> (filler), 1 = <unk> (word not in vocabulary).
    """
    def __init__(self, max_vocab_size=20000, max_length=200):
        self.max_vocab_size = max_vocab_size   # keep at most this many distinct words
        self.max_length = max_length           # every message becomes exactly this many ids
        self.word2idx = {"<pad>": 0, "<unk>": 1}   # word -> id
        self.idx2word = {0: "<pad>", 1: "<unk>"}   # id -> word (reverse lookup)

    def clean_text(self, text):
        """Lowercase the text and split it into a list of words.

        Regex r'\\b\\w+\\b' matches runs of letters/digits/underscore, so punctuation
        is dropped: "FREE prize!!" -> ['free', 'prize'].
        (Despite the name it returns TOKENS, not a cleaned string.)
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit_on_texts(self, texts):
        """Build the vocabulary from the TRAINING texts only.

        Count every word, keep the most frequent (max_vocab_size - 2, because ids 0 and 1 are
        reserved) and give them ids starting from 2. Fitting on training data only
        avoids leaking information from validation/test sets.
        """
        word_counts = Counter()
        for text in texts:
            tokens = self.clean_text(text)
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_sequence(self, text):
        """Convert one message into a list of EXACTLY max_length integers."""
        tokens = self.clean_text(text)
        # .get(w, 1): unknown words (not in vocabulary) become the <unk> id 1
        seq = [self.word2idx.get(w, 1) for w in tokens]
        # Neural nets are fed fixed-size batches, so:
        #   shorter than max_length -> PAD with zeros at the END ("post-padding")
        #   longer than max_length  -> TRUNCATE (cut off the rest of the message)
        if len(seq) < self.max_length:
            seq = seq + [0] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        return seq

    def save(self, filepath):
        """Write the vocabulary to JSON so the API can reload the SAME mapping later."""
        with open(filepath, 'w') as f:
            json.dump({
                "max_vocab_size": self.max_vocab_size,
                "max_length": self.max_length,
                "word2idx": self.word2idx
            }, f)

    @classmethod
    def load(cls, filepath):
        """Re-create a tokenizer from a saved JSON file (a classmethod = alternative constructor)."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        tok = cls(max_vocab_size=data["max_vocab_size"], max_length=data["max_length"])
        tok.word2idx = data["word2idx"]
        # rebuild the reverse mapping (JSON stores keys as strings, ids must be ints)
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        return tok


class TextDataset(Dataset):
    """PyTorch Dataset: tells a DataLoader how many samples exist and how to fetch sample i."""
    def __init__(self, texts, labels, tokenizer):
        self.labels = labels
        # Convert ALL texts to id-sequences once, up front (faster than doing it every epoch)
        self.sequences = [tokenizer.text_to_sequence(t) for t in texts]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Returns (input ids as integer tensor, label as float tensor).
        # Label is float because the loss BCEWithLogitsLoss expects float targets.
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class ScamBiLSTM(nn.Module):
    """Bidirectional 2-layer LSTM text classifier. Output = ONE score (logit) per message.

    Data flow for a batch of shape (batch, 200):
        word ids -> Embedding -> (batch, 200, 128)
                 -> BiLSTM    -> final hidden states, forward + backward joined -> (batch, 256)
                 -> Linear(256->64) -> ReLU -> Dropout -> Linear(64->1) -> logit (batch,)
    A positive logit leans 'scam', negative leans 'safe'; sigmoid(logit) = probability.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super(ScamBiLSTM, self).__init__()
        # EMBEDDING: a learnable lookup table, one 128-number vector per word id.
        # Similar words end up with similar vectors during training.
        # padding_idx=0 -> the <pad> vector stays all zeros and is not learned.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # LSTM = recurrent layer with gates that decide what to remember/forget, so it
        # can keep information from earlier words. bidirectional=True runs one LSTM
        # left->right and another right->left, so each word sees context on both sides.
        # batch_first=True means input shape is (batch, sequence, features).
        # Dropout between the 2 LSTM layers is only valid when there is more than 1 layer.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # hidden_dim * 2 because forward and backward states are concatenated
        self.fc1 = nn.Linear(hidden_dim * 2, 64)    # fully connected layer: 256 -> 64
        self.relu = nn.ReLU()                       # non-linearity: negative values -> 0
        # Dropout randomly zeroes 30% of values during training only, to reduce overfitting
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)                 # final layer: 64 -> 1 score

    def forward(self, x):
        """Run the network. x: LongTensor (batch, seq_len) of word ids."""
        embedded = self.embedding(x)
        # lstm_out = outputs at every position; hn = FINAL hidden state of every layer/direction;
        # cn = final cell state (unused).
        lstm_out, (hn, cn) = self.lstm(embedded)

        # hn has shape (num_layers*2, batch, hidden). The last two entries are the TOP layer:
        # hn[-2] = forward direction, hn[-1] = backward direction.
        # We join them to get one 256-number summary of the whole message.
        # (Because padding is at the END, the forward pass also runs over pad tokens
        #  before producing its final state - a known minor weakness of this design.)
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        # squeeze(-1) turns shape (batch, 1) into (batch,)
        return logits.squeeze(-1)
