import operator
from functools import reduce

import numpy as np

from . import ndarray_backend_numpy

try:
    import ndarray_backend as ndarray_backend_metal  # ty: ignore[unresolved-import]
except ImportError:
    ndarray_backend_metal = None


def prod(values):
    return reduce(operator.mul, values, 1)


class BackendDevice:
    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return isinstance(other, BackendDevice) and self.name == other.name

    def __repr__(self):
        return f"{self.name}()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, indices, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[indices], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def metal():
    return BackendDevice("metal", ndarray_backend_metal)


def numpy():
    return BackendDevice("numpy", ndarray_backend_numpy)


def default_device():
    device = metal()
    return device if device.enabled() else numpy()


class NDArray:
    def __init__(self, other, device=None):
        if isinstance(other, NDArray):
            device = other.device if device is None else device
            self._init(other.to(device) + 0.0)
        elif isinstance(other, np.ndarray):
            device = default_device() if device is None else device
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            self._init(NDArray(np.array(other), device=device))

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        stride = 1
        strides = []
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = (
            NDArray.compact_strides(shape) if strides is None else tuple(strides)
        )
        array._offset = offset
        array._device = default_device() if device is None else device
        array._handle = array.device.Array(prod(shape)) if handle is None else handle
        return array

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return "float32"

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    @property
    def flat(self):
        return self.reshape((self.size,))

    def __repr__(self):
        return f"NDArray({self.numpy()}, device={self.device})"

    def __str__(self):
        return str(self.numpy())

    def fill(self, value):
        self.device.fill(self._handle, value)

    def to(self, device):
        if device == self.device:
            return self
        return NDArray(self.numpy(), device=device)

    def numpy(self):
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        return (
            self.strides == self.compact_strides(self.shape)
            and self.size == self._handle.size
        )

    def compact(self):
        if self.is_compact():
            return self
        out = NDArray.make(self.shape, device=self.device)
        self.device.compact(
            self._handle, out._handle, self.shape, self.strides, self._offset
        )
        return out

    def as_strided(self, shape, strides):
        assert len(shape) == len(strides)
        return NDArray.make(
            shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset,
        )

    def reshape(self, new_shape):
        assert self.size == prod(new_shape)
        compact = self.compact()
        return NDArray.make(
            new_shape,
            device=compact.device,
            handle=compact._handle,
            offset=compact._offset,
        )

    def permute(self, new_axes):
        return NDArray.make(
            tuple(self.shape[i] for i in new_axes),
            strides=tuple(self.strides[i] for i in new_axes),
            device=self.device,
            handle=self._handle,
            offset=self._offset,
        )

    def broadcast_to(self, new_shape):
        assert len(new_shape) == self.ndim
        new_strides = []
        for old, new, stride in zip(self.shape, new_shape, self.strides):
            if old == new:
                new_strides.append(stride)
            elif old == 1:
                new_strides.append(0)
            else:
                raise ValueError("Cannot broadcast dimension")
        return NDArray.make(
            new_shape,
            strides=tuple(new_strides),
            device=self.device,
            handle=self._handle,
            offset=self._offset,
        )

    def process_slice(self, sl, dim):
        start = 0 if sl.start is None else sl.start
        stop = self.shape[dim] if sl.stop is None else sl.stop
        step = 1 if sl.step is None else sl.step
        if start < 0:
            start += self.shape[dim]
        if stop < 0:
            stop += self.shape[dim]
        assert stop > start, "Start must be less than stop"
        assert step > 0, "Negative strides are not supported"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            self.process_slice(idx, i)
            if isinstance(idx, slice)
            else slice(idx, idx + 1, 1)
            for i, idx in enumerate(idxs)
        )
        assert len(idxs) == self.ndim, "Need one index per dimension"

        offset = self._offset
        shape = []
        strides = []
        for dim, idx in enumerate(idxs):
            offset += idx.start * self.strides[dim]
            shape.append(len(range(idx.start, idx.stop, idx.step)))
            strides.append(self.strides[dim] * idx.step)
        return NDArray.make(
            tuple(shape),
            strides=tuple(strides),
            device=self.device,
            handle=self._handle,
            offset=offset,
        )

    def __setitem__(self, idxs, other):
        view = self[idxs]
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    def gelu(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_gelu(self.compact()._handle, out._handle)
        return out

    def __matmul__(self, other):
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]
        rows, inner, cols = self.shape[0], self.shape[1], other.shape[1]
        out = NDArray.make((rows, cols), device=self.device)
        self.device.matmul(
            self.compact()._handle,
            other.compact()._handle,
            out._handle,
            rows,
            inner,
            cols,
        )
        return out

    def batched_matmul(self, other):
        assert self.ndim == 3 and other.ndim == 3
        assert self.shape[0] == other.shape[0]
        assert self.shape[2] == other.shape[1]
        batch, rows, inner, cols = (
            self.shape[0],
            self.shape[1],
            self.shape[2],
            other.shape[2],
        )
        out = NDArray.make((batch, rows, cols), device=self.device)
        self.device.batched_matmul(
            self.compact()._handle,
            other.compact()._handle,
            out._handle,
            batch,
            rows,
            inner,
            cols,
        )
        return out

    def logsumexp_last_axis(self):
        assert self.ndim >= 1
        cols = self.shape[-1]
        rows = self.size // cols
        out = NDArray.make(self.shape[:-1], device=self.device)
        self.device.logsumexp_last_axis(self.compact()._handle, out._handle, cols, rows)
        return out

    def softmax_last_axis(self):
        assert self.ndim >= 1
        cols = self.shape[-1]
        rows = self.size // cols
        out = NDArray.make(self.shape, device=self.device)
        self.device.softmax_last_axis(self.compact()._handle, out._handle, cols, rows)
        return out

    def causal_softmax(self):
        assert self.ndim >= 2
        seq_len = self.shape[-1]
        assert self.shape[-2] == seq_len
        rows = self.size // seq_len
        out = NDArray.make(self.shape, device=self.device)
        self.device.causal_softmax(self.compact()._handle, out._handle, seq_len, rows)
        return out

    def dropout_apply(self, mask, keep_prob):
        assert self.shape == mask.shape
        out = NDArray.make(self.shape, device=self.device)
        self.device.dropout_apply(
            self.compact()._handle,
            mask.compact()._handle,
            out._handle,
            keep_prob,
        )
        return out

    def embedding(self, indices):
        assert self.ndim == 2
        count = indices.size
        dim = self.shape[1]
        out = NDArray.make((*indices.shape, dim), device=self.device)
        self.device.embedding(
            self.compact()._handle,
            indices.compact()._handle,
            out._handle,
            count,
            dim,
        )
        return out

    def reduce_view_out(self, axis, keepdims=False):
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (self.size,))
            out_shape = (1,) * (self.ndim if keepdims else 1)
        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only one reduction axis is supported"
                axis = axis[0]
            view = self.permute(
                tuple(i for i in range(self.ndim) if i != axis) + (axis,)
            )
            out_shape = (
                tuple(1 if i == axis else s for i, s in enumerate(self.shape))
                if keepdims
                else tuple(s for i, s in enumerate(self.shape) if i != axis)
            )
        return view, NDArray.make(out_shape, device=self.device)

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes):
        offset = self._offset
        strides = list(self.strides)
        for axis in axes:
            offset += (self.shape[axis] - 1) * self.strides[axis]
            strides[axis] = -strides[axis]
        return NDArray.make(
            self.shape,
            strides=tuple(strides),
            device=self.device,
            handle=self._handle,
            offset=offset,
        ).compact()

    def pad(self, axes):
        shape = list(self.shape)
        out_slice = []
        for axis, pad in enumerate(axes):
            shape[axis] += sum(pad)
            out_slice.append(slice(pad[0], shape[axis] - pad[1], 1))
        out = full(shape, 0.0, device=self.device)
        out[tuple(out_slice)] = self
        return out


def array(a, dtype="float32", device=None):
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = default_device() if device is None else device
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = default_device() if device is None else device
    return device.full(shape, fill_value, dtype)


def broadcast_to(arr, new_shape):
    return arr.broadcast_to(new_shape)


def reshape(arr, new_shape):
    return arr.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def gelu(a):
    return a.gelu()


def batched_matmul(a, b):
    return a.batched_matmul(b)


def logsumexp_last_axis(a):
    return a.logsumexp_last_axis()


def softmax_last_axis(a):
    return a.softmax_last_axis()


def causal_softmax(a):
    return a.causal_softmax()


def dropout_apply(a, mask, keep_prob):
    return a.dropout_apply(mask, keep_prob)


def embedding(weight, indices):
    return weight.embedding(indices)


def flip(a, axes):
    return a.flip(axes)


def summation(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)
