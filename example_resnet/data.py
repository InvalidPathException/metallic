from __future__ import annotations

import gzip
import os
import pickle
import struct
import sys
import tarfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tensor import Tensor

CIFAR10_PYTHON_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR10_FOLDER_NAME = "cifar-10-batches-py"


def _cifar10_batches_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(n.startswith("data_batch") for n in names) and any(
        n.startswith("test_batch") for n in names
    )


def ensure_cifar10(example_dir: str | os.PathLike[str]) -> Path:
    base = Path(example_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    data_dir = base / CIFAR10_FOLDER_NAME
    if _cifar10_batches_ready(data_dir):
        return data_dir

    archive = base / CIFAR10_ARCHIVE_NAME
    if not archive.is_file():
        with urllib.request.urlopen(CIFAR10_PYTHON_URL) as response:
            archive.write_bytes(response.read())

    with tarfile.open(archive, "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(base, filter="data")
        else:
            tar.extractall(base)

    if not _cifar10_batches_ready(data_dir):
        raise RuntimeError(
            f"After extracting {archive}, expected CIFAR-10 batches in {data_dir}"
        )
    return data_dir


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        image = x
        if np.random.rand() >= self.p:
            return image
        return image[:, :, ::-1] if image.ndim == 3 else image[:, ::-1]


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, x):
        image = x
        shift_h, shift_w = np.random.randint(
            low=-self.padding,
            high=self.padding + 1,
            size=2,
        )
        if image.ndim == 3:
            padded = np.pad(
                image,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            )
            start_h = self.padding + shift_h
            start_w = self.padding + shift_w
            return padded[
                :,
                start_h : start_h + image.shape[1],
                start_w : start_w + image.shape[2],
            ]

        padded = np.pad(image, ((self.padding, self.padding),) * 2)
        start_h = self.padding + shift_h
        start_w = self.padding + shift_w
        return padded[
            start_h : start_h + image.shape[0], start_w : start_w + image.shape[1]
        ]


class Dataset:
    def __init__(self, transforms: Iterable[Transform] | None = None):
        self.transforms = tuple(transforms or ())

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def apply_transforms(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size=1,
        shuffle=False,
        device=None,
        dtype="float32",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.dtype = dtype
        self.ordering = []
        self.index = 0

    def __iter__(self):
        order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(order)
        self.ordering = np.array_split(
            order,
            range(self.batch_size, len(self.dataset), self.batch_size),
        )
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.ordering):
            raise StopIteration
        batch = self.dataset[self.ordering[self.index]]
        self.index += 1
        return tuple(
            Tensor(item, device=self.device, dtype=self.dtype, requires_grad=False)
            for item in batch
        )


class MNISTDataset(Dataset):
    def __init__(self, image_filename, label_filename, transforms=None):
        self.images, self.labels = self._parse_mnist(image_filename, label_filename)
        super().__init__(transforms)

    def __getitem__(self, index):
        images = self.images[index].reshape((-1, 1, 28, 28))
        images = np.stack([self.apply_transforms(image) for image in images])
        if isinstance(index, int):
            images = images[0]
        return images, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _parse_mnist(self, image_filename, label_filename):
        with gzip.open(label_filename, "rb") as label_file:
            _, _ = struct.unpack(">II", label_file.read(8))
            labels = np.frombuffer(label_file.read(), dtype=np.uint8)

        with gzip.open(image_filename, "rb") as image_file:
            _, count, rows, cols = struct.unpack(">IIII", image_file.read(16))
            images = np.frombuffer(image_file.read(), dtype=np.uint8)
            images = images.reshape((count, rows * cols)).astype("float32") / 255.0

        return images, labels.astype("float32")


class CIFAR10Dataset(Dataset):
    def __init__(self, base_folder, train=True, transforms=None):
        super().__init__(transforms)
        images = []
        labels = []
        for filename in sorted(os.listdir(base_folder)):
            is_train_batch = filename.startswith("data_batch")
            is_test_batch = filename.startswith("test_batch")
            if (train and is_train_batch) or (not train and is_test_batch):
                with open(os.path.join(base_folder, filename), "rb") as file:
                    batch = pickle.load(file, encoding="bytes")
                images.append(batch[b"data"])
                labels.extend(batch[b"labels"])

        if not images:
            split = "train" if train else "test"
            raise FileNotFoundError(
                f"No CIFAR-10 {split} batches found in {base_folder}"
            )

        self.images = np.concatenate(images, axis=0).astype("float32") / 255.0
        self.labels = np.array(labels, dtype="float32")

    def __getitem__(self, index):
        images = self.images[index].reshape((-1, 3, 32, 32))
        images = np.stack([self.apply_transforms(image) for image in images])
        if isinstance(index, int):
            images = images[0]
        return images, self.labels[index]

    def __len__(self):
        return len(self.labels)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays, transforms=None):
        super().__init__(transforms)
        self.arrays = arrays

    def __len__(self):
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        values = tuple(array[index] for array in self.arrays)
        if self.transforms and values:
            first = values[0]
            if isinstance(index, int):
                first = self.apply_transforms(first)
            else:
                first = np.stack([self.apply_transforms(item) for item in first])
            values = (first, *values[1:])
        return values


__all__ = [
    "CIFAR10Dataset",
    "DataLoader",
    "ensure_cifar10",
    "Dataset",
    "MNISTDataset",
    "NDArrayDataset",
    "RandomCrop",
    "RandomFlipHorizontal",
    "Transform",
]
