from __future__ import annotations

from collections.abc import Iterable

import init
import ops
from tensor import Tensor


class Parameter(Tensor):
    pass


def _unpack_params(value) -> list[Parameter]:
    if isinstance(value, Parameter):
        return [value] if value.requires_grad else []
    if isinstance(value, Module):
        return value.parameters()
    if isinstance(value, dict):
        return [param for item in value.values() for param in _unpack_params(item)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [param for item in value for param in _unpack_params(item)]
    return []


def _child_modules(value) -> list[Module]:
    if isinstance(value, Module):
        return [value, *_child_modules(value.__dict__)]
    if isinstance(value, dict):
        return [module for item in value.values() for module in _child_modules(item)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [module for item in value for module in _child_modules(item)]
    return []


def _bias(shape, enabled, *, device=None, dtype="float32"):
    return Parameter(
        init.zeros(*shape, device=device, dtype=dtype),
        device=device,
        dtype=dtype,
        requires_grad=enabled,
    )


class Module:
    training: bool

    def __init__(self):
        self.training = True

    def parameters(self) -> list[Parameter]:
        return _unpack_params(self.__dict__)

    def _children(self) -> list[Module]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for module in self._children():
            module.training = False

    def train(self):
        self.training = True
        for module in self._children():
            module.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            init.kaiming_uniform(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
            )
        )
        self.bias = _bias((out_features,), bias, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim

        out = x.reshape((batch_size, self.in_features)) @ self.weight
        out = out.reshape((*batch_shape, self.out_features))
        bias = self.bias.reshape((1,) * len(batch_shape) + (self.out_features,))
        return out + bias.broadcast_to(out.shape)


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        size = 1
        for dim in x.shape[1:]:
            size *= dim
        return x.reshape((x.shape[0], size))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        value = ops.exp(-x)
        return value / (1 + value)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch_size = logits.shape[0]
        y_one_hot = init.one_hot(
            logits.shape[-1],
            y,
            device=logits.device,
            dtype=logits.dtype,
        )
        logsum = ops.logsumexp(logits, axes=1)
        target = ops.summation(logits * y_one_hot, axes=1)
        return ops.summation(logsum - target) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = _bias((dim,), True, device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.sum(axes=0) / x.shape[0]
            centered = x - mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = (centered**2).sum(axes=0) / x.shape[0]
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean
            ).detach()
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            ).detach()
        else:
            mean = self.running_mean
            var = self.running_var

        mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
        var = var.reshape((1, self.dim)).broadcast_to(x.shape)
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias


class BatchNorm2d(BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        x = x.transpose((1, 2)).transpose((2, 3))
        x = x.reshape((batch * height * width, channels))
        y = super().forward(x).reshape((batch, height, width, channels))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = _bias((dim,), True, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.dim
        prefix = x.shape[:-1]
        keep_shape = (*prefix, 1)
        param_shape = (1,) * len(prefix) + (self.dim,)

        mean = (x.sum(axes=len(x.shape) - 1) / self.dim).reshape(keep_shape)
        mean = mean.broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(axes=len(x.shape) - 1).reshape(keep_shape)
        var = (var / self.dim).broadcast_to(x.shape)
        weight = self.weight.reshape(param_shape).broadcast_to(x.shape)
        bias = self.bias.reshape(param_shape).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
        return x * mask / (1 - self.p)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Conv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        stride = stride[0] if isinstance(stride, tuple) else stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in,
                fan_out,
                shape=weight_shape,
                device=device,
                dtype=dtype,
            )
        )
        self.bias = _bias((out_channels,), bias, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose((1, 2)).transpose((2, 3))
        padding = self.kernel_size // 2
        out = ops.conv(x, self.weight, self.stride, padding)
        bias = self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        out = out + bias
        return out.transpose((1, 3)).transpose((2, 3))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        seq_len, batch = x.shape
        one_hot = init.one_hot(
            self.num_embeddings,
            x.reshape((seq_len * batch,)),
            device=self.device,
            dtype=self.dtype,
        )
        return (one_hot @ self.weight).reshape((seq_len, batch, self.embedding_dim))


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        bound = hidden_size**-0.5
        self.weight_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.weight_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.bias_ih = Parameter(
            init.rand(
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            ),
            requires_grad=bias,
        )
        self.bias_hh = Parameter(
            init.rand(
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            ),
            requires_grad=bias,
        )
        if not bias:
            self.bias_ih.data = init.zeros(hidden_size, device=device, dtype=dtype)
            self.bias_hh.data = init.zeros(hidden_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        batch_size = x.shape[0]
        if h is None:
            h = init.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        out = x @ self.weight_ih + h @ self.weight_hh
        bias_ih = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        bias_hh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        out = out + bias_ih + bias_hh
        return ops.relu(out) if self.nonlinearity == "relu" else ops.tanh(out)


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.layers = [
            RNNCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            for layer in range(num_layers)
        ]

    def forward(self, x: Tensor, h0: Tensor | None = None):
        seq_len, batch_size, _ = x.shape
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        inputs = tuple(ops.split(x, axis=0))
        hidden = list(ops.split(h0, axis=0))
        outputs = []
        for step in range(seq_len):
            layer_input = inputs[step]
            for layer, cell in enumerate(self.layers):
                layer_input = cell(layer_input, hidden[layer])
                hidden[layer] = layer_input
            outputs.append(layer_input)
        return ops.stack(outputs, axis=0), ops.stack(hidden, axis=0)


class LSTMCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        bound = hidden_size**-0.5
        gate_size = 4 * hidden_size
        self.weight_ih = Parameter(
            init.rand(
                input_size,
                gate_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.weight_hh = Parameter(
            init.rand(
                hidden_size,
                gate_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.bias_ih = Parameter(
            init.rand(
                gate_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            ),
            requires_grad=bias,
        )
        self.bias_hh = Parameter(
            init.rand(
                gate_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            ),
            requires_grad=bias,
        )
        if not bias:
            self.bias_ih.data = init.zeros(gate_size, device=device, dtype=dtype)
            self.bias_hh.data = init.zeros(gate_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None = None):
        batch_size = x.shape[0]
        if state is None:
            h = init.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
            c = init.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h, c = state

        gates = x @ self.weight_ih + h @ self.weight_hh
        bias_ih = self.bias_ih.reshape((1, 4 * self.hidden_size))
        bias_hh = self.bias_hh.reshape((1, 4 * self.hidden_size))
        gates = gates + bias_ih.broadcast_to(gates.shape)
        gates = gates + bias_hh.broadcast_to(gates.shape)
        gates = gates.reshape((batch_size, 4, self.hidden_size))
        input_gate, forget_gate, cell_gate, output_gate = tuple(
            ops.split(gates, axis=1)
        )

        input_gate = Sigmoid()(input_gate)
        forget_gate = Sigmoid()(forget_gate)
        cell_gate = ops.tanh(cell_gate)
        output_gate = Sigmoid()(output_gate)
        next_c = forget_gate * c + input_gate * cell_gate
        next_h = output_gate * ops.tanh(next_c)
        return next_h, next_c


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.layers = [
            LSTMCell(
                input_size if layer == 0 else hidden_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            for layer in range(num_layers)
        ]

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None = None):
        seq_len, batch_size, _ = x.shape
        if state is None:
            h0 = init.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
            c0 = init.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h0, c0 = state

        inputs = tuple(ops.split(x, axis=0))
        hidden = list(ops.split(h0, axis=0))
        cells = list(ops.split(c0, axis=0))
        outputs = []
        for step in range(seq_len):
            layer_input = inputs[step]
            for layer, cell in enumerate(self.layers):
                next_h, next_c = cell(layer_input, (hidden[layer], cells[layer]))
                hidden[layer] = next_h
                cells[layer] = next_c
                layer_input = next_h
            outputs.append(layer_input)
        return ops.stack(outputs, axis=0), (
            ops.stack(hidden, axis=0),
            ops.stack(cells, axis=0),
        )


class LanguageModel(Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if seq_model == "rnn":
            self.sequence = RNN(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        elif seq_model == "lstm":
            self.sequence = LSTM(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError("seq_model must be 'rnn' or 'lstm'")

        self.embedding = Embedding(
            output_size, embedding_size, device=device, dtype=dtype
        )
        self.linear = Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, state=None):
        embedded = self.embedding(x)
        output, next_state = self.sequence(embedded, state)
        seq_len, batch_size, hidden_size = output.shape
        logits = self.linear(output.reshape((seq_len * batch_size, hidden_size)))
        return logits, next_state


__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv",
    "Dropout",
    "Embedding",
    "Flatten",
    "Identity",
    "LSTM",
    "LSTMCell",
    "LanguageModel",
    "LayerNorm1d",
    "Linear",
    "Module",
    "Parameter",
    "RNN",
    "RNNCell",
    "ReLU",
    "Residual",
    "Sequential",
    "Sigmoid",
    "SoftmaxLoss",
    "Tanh",
]
