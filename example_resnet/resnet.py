from __future__ import annotations

import numpy as np

import nn
from optim import AdamW
from tensor import Tensor

from . import data


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        def conv_bn(in_channels, out_channels, kernel_size, stride):
            return nn.Sequential(
                nn.Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    device=device,
                    dtype=dtype,
                ),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
                nn.ReLU(),
            )

        self.net = nn.Sequential(
            conv_bn(3, 16, 7, 4),
            conv_bn(16, 32, 3, 2),
            nn.Residual(
                nn.Sequential(
                    conv_bn(32, 32, 3, 1),
                    conv_bn(32, 32, 3, 1),
                )
            ),
            conv_bn(32, 64, 3, 2),
            conv_bn(64, 128, 3, 2),
            nn.Residual(
                nn.Sequential(
                    conv_bn(128, 128, 3, 1),
                    conv_bn(128, 128, 3, 1),
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def synthetic_batch(batch_size=4, device=None, dtype="float32", seed=0):
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(batch_size, 3, 32, 32)).astype(dtype)
    labels = rng.integers(0, 10, size=(batch_size,)).astype(dtype)
    return (
        Tensor(images, device=device, dtype=dtype, requires_grad=False),
        Tensor(labels, device=device, dtype=dtype, requires_grad=False),
    )


def accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = np.argmax(logits.detach().numpy(), axis=1)
    return float(np.mean(predictions == labels.numpy().astype("int32")))


def epoch(dataloader, model, loss_fn=None, opt=None):
    loss_fn = nn.SoftmaxLoss() if loss_fn is None else loss_fn
    if opt is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for x, y in dataloader:
        logits = model(x)
        loss = loss_fn(logits, y)
        batch_size = x.shape[0]
        total_loss += float(loss.detach().numpy()) * batch_size
        total_correct += int(
            np.sum(
                np.argmax(logits.detach().numpy(), axis=1) == y.numpy().astype("int32")
            )
        )
        total_examples += batch_size

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

    return total_correct / total_examples, total_loss / total_examples


def cifar10_loader(
    data_dir,
    *,
    train=True,
    batch_size=128,
    device=None,
    dtype="float32",
    augment=False,
):
    transforms = (
        [data.RandomCrop(4), data.RandomFlipHorizontal()] if train and augment else None
    )
    dataset = data.CIFAR10Dataset(data_dir, train=train, transforms=transforms)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        device=device,
        dtype=dtype,
    )


def train_cifar10(
    data_dir,
    *,
    device=None,
    batch_size=128,
    epochs=1,
    lr=1e-3,
    weight_decay=1e-4,
    augment=False,
):
    model = ResNet9(device=device)
    loader = cifar10_loader(
        data_dir,
        train=True,
        batch_size=batch_size,
        device=device,
        augment=augment,
    )
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    last = None
    for _ in range(epochs):
        last = epoch(loader, model, opt=opt)
    return model, last


def smoke_train_step(device=None, batch_size=4, seed=0):
    model = ResNet9(device=device)
    loss_fn = nn.SoftmaxLoss()
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, y = synthetic_batch(batch_size=batch_size, device=device, seed=seed)

    logits = model(x)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    return {
        "device": x.device.name,
        "logits_shape": logits.shape,
        "loss": float(loss.numpy()),
        "accuracy": accuracy(logits, y),
        "parameters": len(model.parameters()),
    }


__all__ = [
    "ResNet9",
    "accuracy",
    "cifar10_loader",
    "epoch",
    "smoke_train_step",
    "synthetic_batch",
    "train_cifar10",
]
