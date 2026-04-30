from __future__ import annotations

from collections.abc import Sequence

from autograd import Tensor, TensorOp
import backend_ndarray as nd


def _normalize_axes(axes, ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(axis if axis >= 0 else axis + ndim for axis in axes)


def _keepdims_shape(shape, axes) -> tuple[int, ...]:
    axes = set(_normalize_axes(axes, len(shape)))
    return tuple(1 if axis in axes else dim for axis, dim in enumerate(shape))


def _sum_to_shape(grad: Tensor, shape: tuple[int, ...]) -> Tensor:
    axes = list(range(len(grad.shape) - len(shape)))
    axes.extend(
        axis + len(grad.shape) - len(shape)
        for axis, dim in enumerate(shape)
        if dim == 1
    )
    if axes:
        grad = summation(grad, tuple(axes))
    return reshape(grad, shape)


class EWiseAdd(TensorOp):
    def compute(self, a, b):
        return a + b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return _sum_to_shape(out_grad, lhs.shape), _sum_to_shape(out_grad, rhs.shape)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a + self.scalar

    def gradient(self, out_grad, node):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a, b):
        return a * b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (
            _sum_to_shape(out_grad * rhs, lhs.shape),
            _sum_to_shape(out_grad * lhs, rhs.shape),
        )


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a * self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (
            _sum_to_shape(out_grad / rhs, lhs.shape),
            _sum_to_shape(-out_grad * lhs / (rhs**2), rhs.shape),
        )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a**self.scalar

    def gradient(self, out_grad, node):
        base = node.inputs[0]
        return out_grad * self.scalar * (base ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        grad_lhs = out_grad @ transpose(rhs)
        grad_rhs = transpose(lhs) @ out_grad
        return _sum_to_shape(grad_lhs, lhs.shape), _sum_to_shape(grad_rhs, rhs.shape)


def matmul(a, b):
    return MatMul()(a, b)


class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        axes = list(range(len(a.shape)))
        if self.axes is None:
            axes[-1], axes[-2] = axes[-2], axes[-1]
        else:
            first, second = self.axes
            axes[first], axes[second] = axes[second], axes[first]
        return a.permute(tuple(axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def compute(self, a):
        return nd.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def compute(self, a):
        return nd.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        return _sum_to_shape(out_grad, node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        axes = _normalize_axes(self.axes, len(a.shape))
        out = a
        for axis in sorted(axes, reverse=True):
            out = out.sum(axis)
        return out

    def gradient(self, out_grad, node):
        shape = _keepdims_shape(node.inputs[0].shape, self.axes)
        return broadcast_to(reshape(out_grad, shape), node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class Log(TensorOp):
    def compute(self, a):
        return nd.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return nd.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return nd.tanh(a)

    def gradient(self, out_grad, node):
        value = tanh(node.inputs[0])
        return out_grad * (1 - value**2)


def tanh(a):
    return Tanh()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return nd.maximum(a, 0)

    def gradient(self, out_grad, node):
        hot = node.inputs[0].realize_cached_data() >= 0
        return out_grad * Tensor(hot, device=out_grad.device, requires_grad=False)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        keep_shape = _keepdims_shape(a.shape, self.axes)
        max_value = a.max(self.axes)
        shifted = a - nd.broadcast_to(nd.reshape(max_value, keep_shape), a.shape)
        return nd.log(nd.exp(shifted).sum(self.axes)) + max_value

    def gradient(self, out_grad, node):
        keep_shape = _keepdims_shape(node.inputs[0].shape, self.axes)
        logsum = broadcast_to(reshape(node, keep_shape), node.inputs[0].shape)
        grad = broadcast_to(reshape(out_grad, keep_shape), node.inputs[0].shape)
        return grad * exp(node.inputs[0] - logsum)


def logsumexp(a, axes=None):
    return LogSumExp(axes)(a)


def stack(tensors: Sequence[Tensor], axis: int = 0):
    raise NotImplementedError("stack will be added with TensorTuple support")


def split(tensor: Tensor, axis: int = 0):
    raise NotImplementedError("split will be added with TensorTuple support")


def dilate(tensor: Tensor, axes: tuple[int, ...], dilation: int):
    raise NotImplementedError("dilate will be added with convolution support")


def undilate(tensor: Tensor, axes: tuple[int, ...], dilation: int):
    raise NotImplementedError("undilate will be added with convolution support")


def conv(a: Tensor, b: Tensor, stride: int = 1, padding: int = 0):
    raise NotImplementedError("conv will be added after the dense tensor core")
