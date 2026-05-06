from __future__ import annotations

import math

import backend_ndarray as nd
from tensor import Tensor


def _device(device):
    return nd.default_device() if device is None else device


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    device = _device(device)
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    device = _device(device)
    array = device.randn(*shape, dtype=dtype) * std + mean
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    device = _device(device)
    array = device.full(shape, c, dtype=dtype)
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(
        *shape,
        c=1.0,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(
        *shape,
        c=0.0,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def randb(*shape, p=0.5, device=None, dtype="float32", requires_grad=False):
    device = _device(device)
    array = device.rand(*shape, dtype="float32") <= p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, indices, device=None, dtype="float32", requires_grad=False):
    device = _device(device)
    return Tensor(
        device.one_hot(n, indices.numpy().astype("int32"), dtype=dtype),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def zeros_like(tensor, *, device=None, requires_grad=False):
    device = tensor.device if device is None else device
    return zeros(
        *tensor.shape,
        device=device,
        dtype=tensor.dtype,
        requires_grad=requires_grad,
    )


def ones_like(tensor, *, device=None, requires_grad=False):
    device = tensor.device if device is None else device
    return ones(
        *tensor.shape,
        device=device,
        dtype=tensor.dtype,
        requires_grad=requires_grad,
    )


def xavier_uniform(
    fan_in,
    fan_out,
    shape=None,
    gain=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    bound = gain * math.sqrt(6 / (fan_in + fan_out))
    shape = (fan_in, fan_out) if shape is None else shape
    return rand(
        *shape,
        low=-bound,
        high=bound,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def xavier_normal(
    fan_in,
    fan_out,
    shape=None,
    gain=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    shape = (fan_in, fan_out) if shape is None else shape
    return randn(
        *shape,
        mean=0.0,
        std=std,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def kaiming_uniform(
    fan_in,
    fan_out,
    shape=None,
    nonlinearity="relu",
    device=None,
    dtype="float32",
    requires_grad=False,
):
    assert nonlinearity == "relu"
    bound = math.sqrt(6 / fan_in)
    shape = (fan_in, fan_out) if shape is None else shape
    return rand(
        *shape,
        low=-bound,
        high=bound,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def kaiming_normal(
    fan_in,
    fan_out,
    shape=None,
    nonlinearity="relu",
    device=None,
    dtype="float32",
    requires_grad=False,
):
    assert nonlinearity == "relu"
    std = math.sqrt(2 / fan_in)
    shape = (fan_in, fan_out) if shape is None else shape
    return randn(
        *shape,
        mean=0.0,
        std=std,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )


__all__ = [
    "constant",
    "kaiming_normal",
    "kaiming_uniform",
    "one_hot",
    "ones",
    "ones_like",
    "rand",
    "randb",
    "randn",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
    "zeros_like",
]
