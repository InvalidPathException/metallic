from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from backend_ndarray import NDArray, array as backend_array, default_device, full


def _ops():
    import ops

    return ops


def _ensure_tensor(value: Any, *, device=None, dtype="float32") -> Tensor:
    return (
        value
        if isinstance(value, Tensor)
        else Tensor(value, device=device, dtype=dtype)
    )


class Op:
    def __call__(self, *args):
        raise NotImplementedError

    def compute(self, *args):
        raise NotImplementedError

    def gradient(self, out_grad: Tensor, node: Value):
        raise NotImplementedError

    def gradient_as_tuple(self, out_grad: Tensor, node: Value) -> tuple[Tensor, ...]:
        grad = self.gradient(out_grad, node)
        if isinstance(grad, tuple):
            return grad
        if isinstance(grad, list):
            return tuple(grad)
        return (grad,)


class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    op: Op | None
    inputs: tuple[Value, ...]
    cached_data: Any
    requires_grad: bool
    grad: Tensor | None

    __hash__ = object.__hash__

    def _init(
        self,
        op: Op | None,
        inputs: Iterable[Value],
        *,
        cached_data=None,
        requires_grad: bool | None = None,
    ):
        self.op = op
        self.inputs = tuple(inputs)
        self.cached_data = cached_data
        self.grad = None
        self.requires_grad = (
            any(input.requires_grad for input in self.inputs)
            if requires_grad is None
            else requires_grad
        )

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(None, (), cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: Iterable[Value]):
        value = cls.__new__(cls)
        value._init(op, inputs)
        if not value.requires_grad:
            return value.detach()
        value.realize_cached_data()
        return value

    def realize_cached_data(self):
        if self.cached_data is None:
            assert self.op is not None
            self.cached_data = self.op.compute(
                *(input.realize_cached_data() for input in self.inputs)
            )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def numpy(self):
        data = self.realize_cached_data()
        return data.numpy() if isinstance(data, NDArray) else np.asarray(data)


class TensorTuple(Value):
    def __len__(self):
        return len(self.realize_cached_data())

    def __getitem__(self, index: int):
        return _ops().tuple_get_item(self, index)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __repr__(self):
        return f"TensorTuple({self.tuple()})"

    def __str__(self):
        return repr(self)

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return _ops().make_tuple(*(self[i] + other[i] for i in range(len(self))))

    def tuple(self):
        return tuple(self)

    def detach(self):
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: Tensor | None

    def __init__(
        self,
        data,
        *,
        device=None,
        dtype="float32",
        requires_grad=True,
    ):
        if isinstance(data, Tensor):
            device = data.device if device is None else device
            dtype = data.dtype if dtype is None else dtype
            cached_data = (
                data.realize_cached_data()
                if device == data.device and dtype == data.dtype
                else self._array_from_numpy(data.numpy(), device=device, dtype=dtype)
            )
        else:
            cached_data = self._array_from_numpy(
                data,
                device=default_device() if device is None else device,
                dtype=dtype,
            )

        self.grad = None
        self._init(None, (), cached_data=cached_data, requires_grad=requires_grad)

    @staticmethod
    def _array_from_numpy(data, *, device, dtype):
        if isinstance(data, NDArray):
            return (
                data.to(device)
                if device is not None and data.device != device
                else data
            )
        return backend_array(np.asarray(data, dtype=dtype), device=device)

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        tensor = cls.__new__(cls)
        tensor._init(
            None,
            (),
            cached_data=data.realize_cached_data()
            if isinstance(data, Tensor)
            else data,
            requires_grad=requires_grad,
        )
        return tensor

    @classmethod
    def make_from_op(cls, op: Op, inputs: Iterable[Value]):
        tensor = cls.__new__(cls)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        value = _ensure_tensor(value, device=self.device, dtype=self.dtype)
        assert value.dtype == self.dtype
        self.cached_data = value.realize_cached_data()

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return self.realize_cached_data().device

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    def backward(self, out_grad=None):
        if out_grad is None:
            out_grad = Tensor(
                full(self.shape, 1.0, device=self.device),
                requires_grad=False,
            )
        elif not isinstance(out_grad, Tensor):
            out_grad = Tensor(
                out_grad, device=self.device, dtype=self.dtype, requires_grad=False
            )
        compute_gradients(self, out_grad)

    def __repr__(self):
        return f"Tensor({self.realize_cached_data()})"

    def __str__(self):
        return str(self.realize_cached_data())

    def __add__(self, other):
        ops = _ops()
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(self, other)
        return ops.AddScalar(other)(self)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        ops = _ops()
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(self, ops.Negate()(other))
        return ops.AddScalar(-other)(self)

    def __rsub__(self, other):
        ops = _ops()
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(other, ops.Negate()(self))
        return ops.AddScalar(other)(ops.Negate()(self))

    def __mul__(self, other):
        ops = _ops()
        if isinstance(other, Tensor):
            return ops.EWiseMul()(self, other)
        return ops.MulScalar(other)(self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        ops = _ops()
        if isinstance(other, Tensor):
            return ops.EWiseDiv()(self, other)
        return ops.DivScalar(other)(self)

    def __rtruediv__(self, other):
        ops = _ops()
        if isinstance(other, Tensor):
            return ops.EWiseDiv()(other, self)
        return ops.EWiseDiv()(
            Tensor(other, device=self.device, dtype=self.dtype, requires_grad=False),
            self,
        )

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise TypeError("Tensor exponents are not supported")
        return _ops().PowerScalar(other)(self)

    def __neg__(self):
        return _ops().Negate()(self)

    def __matmul__(self, other):
        other = _ensure_tensor(other, device=self.device, dtype=self.dtype)
        return _ops().MatMul()(self, other)

    def __rmatmul__(self, other):
        other = _ensure_tensor(other, device=self.device, dtype=self.dtype)
        return _ops().MatMul()(other, self)

    def sum(self, axes=None):
        return _ops().Summation(axes)(self)

    def broadcast_to(self, shape):
        return _ops().BroadcastTo(shape)(self)

    def reshape(self, shape):
        return _ops().Reshape(shape)(self)

    def transpose(self, axes=None):
        return _ops().Transpose(axes)(self)


def tensor(data, *, device=None, dtype="float32", requires_grad=True):
    return Tensor(data, device=device, dtype=dtype, requires_grad=requires_grad)


def compute_gradients(output: Tensor, out_grad: Tensor):
    node_to_grads: dict[Value, list[Tensor]] = {output: [out_grad]}

    for node in reversed(find_topo_sort([output])):
        grad = sum_tensors(node_to_grads[node])
        node.grad = grad

        if node.is_leaf():
            continue

        assert node.op is not None
        for input_node, input_grad in zip(
            node.inputs,
            node.op.gradient_as_tuple(grad, node),
        ):
            if input_grad is None:
                continue
            node_to_grads.setdefault(input_node, []).append(input_grad)


def find_topo_sort(nodes: Iterable[Value]) -> list[Value]:
    topo = []
    visited = set()
    for node in nodes:
        topo_sort_dfs(node, visited, topo)
    return topo


def topo_sort_dfs(node: Value, visited: set[Value], topo: list[Value]):
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo)
    topo.append(node)


def sum_tensors(tensors: list[Tensor]):
    assert tensors
    total = tensors[0]
    for tensor in tensors[1:]:
        total = total + tensor
    return total
