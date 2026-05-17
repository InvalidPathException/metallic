from __future__ import annotations

import math
from collections.abc import Sequence

import backend_ndarray as nd
from tensor import Tensor, TensorOp, TensorTuple, TensorTupleOp


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args):
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(out_grad[index] for index in range(len(out_grad)))


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index: int):
        self.index = index

    def __call__(self, *args):
        value = args[0]
        fold_const = args[1] if len(args) > 1 else True
        assert isinstance(value, TensorTuple)
        if fold_const and isinstance(value.op, MakeTensorTuple):
            return value.inputs[self.index]
        return Tensor.make_from_op(self, [value])

    def compute(self, *args):
        (value,) = args
        return value[self.index]

    def gradient(self, out_grad, node):
        values = node.inputs[0]
        return make_tuple(
            *(
                out_grad
                if index == self.index
                else Tensor(nd.full(values[index].shape, 0, device=out_grad.device))
                for index in range(len(values))
            )
        )


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


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
    def compute(self, *args):
        a, b = args
        return a + b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return _sum_to_shape(out_grad, lhs.shape), _sum_to_shape(out_grad, rhs.shape)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, *args):
        (a,) = args
        return a + self.scalar

    def gradient(self, out_grad, node):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, *args):
        a, b = args
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

    def compute(self, *args):
        (a,) = args
        return a * self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, *args):
        a, b = args
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

    def compute(self, *args):
        (a,) = args
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, *args):
        (a,) = args
        return a**self.scalar

    def gradient(self, out_grad, node):
        base = node.inputs[0]
        return out_grad * self.scalar * (base ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class Negate(TensorOp):
    def compute(self, *args):
        (a,) = args
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class MatMul(TensorOp):
    def compute(self, *args):
        a, b = args
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        grad_lhs = out_grad @ transpose(rhs)
        grad_rhs = transpose(lhs) @ out_grad
        return _sum_to_shape(grad_lhs, lhs.shape), _sum_to_shape(grad_rhs, rhs.shape)


def matmul(a, b):
    return MatMul()(a, b)


def _transpose_last_two(tensor: Tensor) -> Tensor:
    ndim = len(tensor.shape)
    return transpose(tensor, (ndim - 2, ndim - 1))


class BatchedMatMul(TensorOp):
    def compute(self, *args):
        a, b = args
        return nd.batched_matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (
            batched_matmul(out_grad, _transpose_last_two(rhs)),
            batched_matmul(_transpose_last_two(lhs), out_grad),
        )


def batched_matmul(a, b):
    return BatchedMatMul()(a, b)


class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, *args):
        (a,) = args
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

    def compute(self, *args):
        (a,) = args
        return nd.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def compute(self, *args):
        (a,) = args
        return nd.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        return _sum_to_shape(out_grad, node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, *args):
        (a,) = args
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
    def compute(self, *args):
        (a,) = args
        return nd.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class Tanh(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.tanh(a)

    def gradient(self, out_grad, node):
        value = tanh(node.inputs[0])
        return out_grad * (1 - value**2)


def tanh(a):
    return Tanh()(a)


class GeLU(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.gelu(a)

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        coeff = math.sqrt(2 / math.pi)
        inner = coeff * (x + 0.044715 * x**3)
        tanh_inner = tanh(inner)
        inner_grad = coeff * (1 + 3 * 0.044715 * x**2)
        grad = 0.5 * (1 + tanh_inner) + 0.5 * x * (1 - tanh_inner**2) * inner_grad
        return out_grad * grad


def gelu(a):
    return GeLU()(a)


class ReLU(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.maximum(a, 0)

    def gradient(self, out_grad, node):
        hot = node.inputs[0].realize_cached_data() >= 0
        return out_grad * Tensor(hot, device=out_grad.device, requires_grad=False)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, *args):
        (a,) = args
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


class LogSumExpLastAxis(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.logsumexp_last_axis(a)

    def gradient(self, out_grad, node):
        value = node.inputs[0]
        keep_shape = (*value.shape[:-1], 1)
        logsum = reshape(node, keep_shape).broadcast_to(value.shape)
        grad = reshape(out_grad, keep_shape).broadcast_to(value.shape)
        return grad * exp(value - logsum)


def logsumexp_last_axis(a):
    return LogSumExpLastAxis()(a)


class SoftmaxLastAxis(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.softmax_last_axis(a)

    def gradient(self, out_grad, node):
        value = node
        correction = summation(out_grad * value, axes=len(value.shape) - 1)
        correction = reshape(correction, (*value.shape[:-1], 1))
        correction = correction.broadcast_to(value.shape)
        return value * (out_grad - correction)


def softmax_last_axis(a):
    return SoftmaxLastAxis()(a)


class CausalSoftmax(TensorOp):
    def compute(self, *args):
        (a,) = args
        return nd.causal_softmax(a)

    def gradient(self, out_grad, node):
        value = node
        correction = summation(out_grad * value, axes=len(value.shape) - 1)
        correction = reshape(correction, (*value.shape[:-1], 1))
        correction = correction.broadcast_to(value.shape)
        return value * (out_grad - correction)


def causal_softmax(a):
    return CausalSoftmax()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, *args):
        (values,) = args
        axis = self.axis if self.axis >= 0 else self.axis + len(values[0].shape) + 1
        count = len(values)
        out = nd.empty((count, values[0].size), device=values[0].device)
        for index, value in enumerate(values):
            out[index, :] = value.reshape((1, value.size))
        shape = (count,) + values[0].shape
        axes = list(range(1, len(shape)))
        axes.insert(axis, 0)
        return out.reshape(shape).permute(tuple(axes))

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(tensors: Sequence[Tensor], axis: int = 0):
    return Stack(axis)(make_tuple(*tensors))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, *args):
        (value,) = args
        axis = self.axis if self.axis >= 0 else self.axis + len(value.shape)
        count = value.shape[axis]
        axes = list(range(len(value.shape)))
        axes[0], axes[axis] = axes[axis], axes[0]
        shape = list(value.shape)
        shape.pop(axis)
        value = value.permute(tuple(axes)).reshape((count, value.size // count))
        return tuple(value[index, :].reshape(tuple(shape)) for index in range(count))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return stack(tuple(out_grad), self.axis)


def split(tensor: Tensor, axis: int = 0):
    return Split(axis)(tensor)


class Flip(TensorOp):
    def __init__(self, axes):
        self.axes = axes

    def compute(self, *args):
        (value,) = args
        axes = tuple(range(len(value.shape))) if self.axes is None else self.axes
        return value.flip(axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(tensor: Tensor, axes=None):
    return Flip(axes)(tensor)


class Dilate(TensorOp):
    def __init__(self, axes: tuple[int, ...], dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, *args):
        (value,) = args
        axes = set(_normalize_axes(self.axes, len(value.shape)))
        shape = tuple(
            dim * (self.dilation + 1) if axis in axes else dim
            for axis, dim in enumerate(value.shape)
        )
        out = nd.full(shape, 0, device=value.device)
        slices = tuple(
            slice(0, dim, self.dilation + 1) if axis in axes else slice(None)
            for axis, dim in enumerate(shape)
        )
        out[slices] = value
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(tensor: Tensor, axes: tuple[int, ...], dilation: int):
    return Dilate(axes, dilation)(tensor)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple[int, ...], dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, *args):
        (value,) = args
        axes = set(_normalize_axes(self.axes, len(value.shape)))
        slices = tuple(
            slice(0, dim, self.dilation + 1) if axis in axes else slice(None)
            for axis, dim in enumerate(value.shape)
        )
        return value[slices]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(tensor: Tensor, axes: tuple[int, ...], dilation: int):
    return UnDilate(axes, dilation)(tensor)


class Conv(TensorOp):
    def __init__(self, stride: int = 1, padding: int = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, *args):
        value, weight = args
        batch, height, width, in_channels = value.shape
        kernel, _, _, out_channels = weight.shape
        out_height = (height + 2 * self.padding - kernel) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel) // self.stride + 1
        out = nd.empty(
            (batch, out_height, out_width, out_channels),
            device=value.device,
        )
        value.device.conv_forward(
            value.compact()._handle,
            weight.compact()._handle,
            out._handle,
            batch,
            height,
            width,
            in_channels,
            kernel,
            out_channels,
            self.stride,
            self.padding,
            out_height,
            out_width,
        )
        return out

    def gradient(self, out_grad, node):
        value, weight = node.inputs
        batch, height, width, in_channels = value.shape
        kernel, _, _, out_channels = weight.shape
        out_height, out_width = out_grad.shape[1], out_grad.shape[2]

        grad_value = nd.empty(value.shape, device=value.device)
        grad_weight = nd.empty(weight.shape, device=weight.device)
        value.device.conv_backward_input(
            out_grad.realize_cached_data().compact()._handle,
            weight.realize_cached_data().compact()._handle,
            grad_value._handle,
            batch,
            height,
            width,
            in_channels,
            kernel,
            out_channels,
            self.stride,
            self.padding,
            out_height,
            out_width,
        )
        value.device.conv_backward_weight(
            value.realize_cached_data().compact()._handle,
            out_grad.realize_cached_data().compact()._handle,
            grad_weight._handle,
            batch,
            height,
            width,
            in_channels,
            kernel,
            out_channels,
            self.stride,
            self.padding,
            out_height,
            out_width,
        )
        return Tensor(grad_value), Tensor(grad_weight)


def conv(a: Tensor, b: Tensor, stride: int = 1, padding: int = 0):
    return Conv(stride, padding)(a, b)


class DropoutApply(TensorOp):
    def __init__(self, keep_prob: float):
        self.keep_prob = keep_prob

    def compute(self, *args):
        value, mask = args
        return nd.dropout_apply(value, mask, self.keep_prob)

    def gradient(self, out_grad, node):
        _, mask = node.inputs
        return dropout_apply(out_grad, mask, self.keep_prob), None


def dropout_apply(value: Tensor, mask: Tensor, keep_prob: float):
    return DropoutApply(keep_prob)(value, mask)


class EmbeddingLookup(TensorOp):
    def compute(self, *args):
        weight, indices = args
        return nd.embedding(weight, indices)

    def gradient(self, out_grad, node):
        weight, indices = node.inputs
        grad = nd.embedding_backward(
            out_grad.realize_cached_data(),
            indices.realize_cached_data(),
            weight.shape,
        )
        return Tensor(grad), None


def embedding(weight: Tensor, indices: Tensor):
    return EmbeddingLookup()(weight, indices)
