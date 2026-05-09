from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from tensor import Tensor

PTB_BASE_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data"
PTB_FILES = ("train.txt", "valid.txt", "test.txt")


def ensure_ptb(example_dir: str | os.PathLike[str]) -> Path:
    data_dir = Path(example_dir).expanduser().resolve() / "ptb"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in PTB_FILES:
        path = data_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(f"{PTB_BASE_URL}/ptb.{filename}", path)
    return data_dir


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, base_dir, max_lines=None):
        self.base_dir = Path(base_dir)
        self.dictionary = Dictionary()
        self.train = self.tokenize(self.base_dir / "train.txt", max_lines)
        self.valid = self.tokenize(self.base_dir / "valid.txt", max_lines)
        self.test = self.tokenize(self.base_dir / "test.txt", max_lines)

    def tokenize(self, path, max_lines=None):
        ids = []
        with open(path) as file:
            lines = file.readlines()[:max_lines]
        for line in lines:
            for word in [*line.split(), "<eos>"]:
                ids.append(self.dictionary.add_word(word))
        return ids


def batchify(data, batch_size, dtype="float32"):
    values = np.array(data, dtype=dtype)
    count = len(values) // batch_size
    values = values[: batch_size * count]
    return values.reshape((batch_size, count)).transpose()


def get_batch(batches, index, seq_len, device=None, dtype="float32"):
    length = min(seq_len, len(batches) - 1 - index)
    x = batches[index : index + length]
    y = batches[index + 1 : index + 1 + length].reshape((-1,))
    return (
        Tensor(x, device=device, dtype=dtype, requires_grad=False),
        Tensor(y, device=device, dtype=dtype, requires_grad=False),
    )


__all__ = [
    "Corpus",
    "Dictionary",
    "batchify",
    "ensure_ptb",
    "get_batch",
]
