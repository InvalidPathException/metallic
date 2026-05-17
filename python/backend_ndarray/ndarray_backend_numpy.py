import numpy as np


__device_name__ = "numpy"
_datatype = np.float32
_dtype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size):
        self.array = np.empty(size, dtype=np.float32)

    @property
    def size(self):
        return self.array.size


def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple(s * _dtype_size for s in strides)
    )


def from_numpy(a, out):
    out.array[:] = a.flatten()


def fill(out, val):
    out.array.fill(val)


def compact(a, out, shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    out.array[:] = np.tanh(a.array)


def ewise_gelu(a, out):
    x = a.array
    out.array[:] = 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def batched_matmul(a, b, out, batch, m, n, p):
    lhs = a.array.reshape(batch, m, n)
    rhs = b.array.reshape(batch, n, p)
    out.array[:] = np.matmul(lhs, rhs).reshape(-1)


def logsumexp_last_axis(a, out, cols, rows):
    values = a.array.reshape(rows, cols)
    max_values = values.max(axis=1, keepdims=True)
    out.array[:] = np.log(np.exp(values - max_values).sum(axis=1)) + max_values[:, 0]


def softmax_last_axis(a, out, cols, rows):
    values = a.array.reshape(rows, cols)
    max_values = values.max(axis=1, keepdims=True)
    exp_values = np.exp(values - max_values)
    out.array[:] = (exp_values / exp_values.sum(axis=1, keepdims=True)).reshape(-1)


def causal_softmax(a, out, seq_len, rows):
    values = a.array.reshape(rows, seq_len)
    result = np.zeros_like(values)
    for row in range(rows):
        query = row % seq_len
        visible = values[row, : query + 1]
        max_value = visible.max()
        exp_values = np.exp(visible - max_value)
        result[row, : query + 1] = exp_values / exp_values.sum()
    out.array[:] = result.reshape(-1)


def dropout_apply(a, mask, out, keep_prob):
    out.array[:] = a.array * mask.array / keep_prob


def embedding(weight, indices, out, count, dim):
    idx = indices.array[:count].astype(np.int32)
    out.array[:] = weight.array.reshape(-1, dim)[idx].reshape(-1)


def embedding_backward(out_grad, indices, weight_grad, count, dim):
    weight_grad.array.fill(0)
    idx = indices.array[:count].astype(np.int32)
    grad = out_grad.array.reshape(count, dim)
    np.add.at(weight_grad.array.reshape(-1, dim), idx, grad)


def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)


def _conv_out_shape(height, width, kernel, stride, padding):
    return (
        (height + 2 * padding - kernel) // stride + 1,
        (width + 2 * padding - kernel) // stride + 1,
    )


def conv_forward(
    x,
    w,
    out,
    n,
    height,
    width,
    in_channels,
    kernel,
    out_channels,
    stride,
    padding,
    out_height,
    out_width,
):
    x_view = x.array.reshape(n, height, width, in_channels)
    w_view = w.array.reshape(kernel, kernel, in_channels, out_channels)
    out_view = out.array.reshape(n, out_height, out_width, out_channels)
    x_pad = np.pad(
        x_view,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    )

    for batch in range(n):
        for oh in range(out_height):
            for ow in range(out_width):
                row = oh * stride
                col = ow * stride
                patch = x_pad[batch, row : row + kernel, col : col + kernel, :]
                out_view[batch, oh, ow, :] = np.sum(
                    patch[:, :, :, None] * w_view,
                    axis=(0, 1, 2),
                )


def conv_backward_input(
    out_grad,
    w,
    x_grad,
    n,
    height,
    width,
    in_channels,
    kernel,
    out_channels,
    stride,
    padding,
    out_height,
    out_width,
):
    out_grad_view = out_grad.array.reshape(n, out_height, out_width, out_channels)
    w_view = w.array.reshape(kernel, kernel, in_channels, out_channels)
    grad_pad = np.zeros(
        (n, height + 2 * padding, width + 2 * padding, in_channels),
        dtype=np.float32,
    )

    for batch in range(n):
        for oh in range(out_height):
            for ow in range(out_width):
                row = oh * stride
                col = ow * stride
                for co in range(out_channels):
                    grad_pad[batch, row : row + kernel, col : col + kernel, :] += (
                        out_grad_view[batch, oh, ow, co] * w_view[:, :, :, co]
                    )

    if padding:
        x_grad.array[:] = grad_pad[:, padding:-padding, padding:-padding, :].reshape(-1)
    else:
        x_grad.array[:] = grad_pad.reshape(-1)


def conv_backward_weight(
    x,
    out_grad,
    w_grad,
    n,
    height,
    width,
    in_channels,
    kernel,
    out_channels,
    stride,
    padding,
    out_height,
    out_width,
):
    x_view = x.array.reshape(n, height, width, in_channels)
    out_grad_view = out_grad.array.reshape(n, out_height, out_width, out_channels)
    x_pad = np.pad(
        x_view,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    )
    grad = np.zeros((kernel, kernel, in_channels, out_channels), dtype=np.float32)

    for batch in range(n):
        for oh in range(out_height):
            for ow in range(out_width):
                row = oh * stride
                col = ow * stride
                patch = x_pad[batch, row : row + kernel, col : col + kernel, :]
                for co in range(out_channels):
                    grad[:, :, :, co] += patch * out_grad_view[batch, oh, ow, co]

    w_grad.array[:] = grad.reshape(-1)
