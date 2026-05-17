#include <metal_stdlib>
using namespace metal;

#define TILE 8
#define MAX_VEC_SIZE 8

struct Vec {
    uint size;
    int data[MAX_VEC_SIZE];
};

inline uint strided_index(uint gid, constant Vec &shape,
                          constant Vec &strides) {
    uint idx = 0;
    uint next = gid;
    for (int i = int(shape.size) - 1; i >= 0; --i) {
        idx += (next % uint(shape.data[i])) * uint(strides.data[i]);
        next /= uint(shape.data[i]);
    }
    return idx;
}

kernel void fill_kernel(device float *out [[buffer(0)]],
                        constant float &val [[buffer(1)]],
                        constant uint &size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = val;
}

kernel void compact_kernel(const device float *a [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant uint &size [[buffer(2)]],
                           constant Vec &shape [[buffer(3)]],
                           constant Vec &strides [[buffer(4)]],
                           constant uint &offset [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[strided_index(gid, shape, strides) + offset];
}

kernel void ewise_setitem_kernel(const device float *a [[buffer(0)]],
                                 device float *out [[buffer(1)]],
                                 constant uint &size [[buffer(2)]],
                                 constant Vec &shape [[buffer(3)]],
                                 constant Vec &strides [[buffer(4)]],
                                 constant uint &offset [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[strided_index(gid, shape, strides) + offset] = a[gid];
}

kernel void scalar_setitem_kernel(device float *out [[buffer(0)]],
                                  constant float &val [[buffer(1)]],
                                  constant uint &size [[buffer(2)]],
                                  constant Vec &shape [[buffer(3)]],
                                  constant Vec &strides [[buffer(4)]],
                                  constant uint &offset [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[strided_index(gid, shape, strides) + offset] = val;
}

kernel void ewise_add(const device float *a [[buffer(0)]],
                      const device float *b [[buffer(1)]],
                      device float *out [[buffer(2)]],
                      constant uint &size [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] + b[gid];
}

kernel void scalar_add(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant float &val [[buffer(2)]],
                       constant uint &size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] + val;
}

kernel void ewise_mul(const device float *a [[buffer(0)]],
                      const device float *b [[buffer(1)]],
                      device float *out [[buffer(2)]],
                      constant uint &size [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] * b[gid];
}

kernel void scalar_mul(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant float &val [[buffer(2)]],
                       constant uint &size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] * val;
}

kernel void ewise_div(const device float *a [[buffer(0)]],
                      const device float *b [[buffer(1)]],
                      device float *out [[buffer(2)]],
                      constant uint &size [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] / b[gid];
}

kernel void scalar_div(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant float &val [[buffer(2)]],
                       constant uint &size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] / val;
}

kernel void scalar_power(const device float *a [[buffer(0)]],
                         device float *out [[buffer(1)]],
                         constant float &val [[buffer(2)]],
                         constant uint &size [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = pow(a[gid], val);
}

kernel void ewise_maximum(const device float *a [[buffer(0)]],
                          const device float *b [[buffer(1)]],
                          device float *out [[buffer(2)]],
                          constant uint &size [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = max(a[gid], b[gid]);
}

kernel void scalar_maximum(const device float *a [[buffer(0)]],
                           device float *out [[buffer(1)]],
                           constant float &val [[buffer(2)]],
                           constant uint &size [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = max(a[gid], val);
}

kernel void ewise_eq(const device float *a [[buffer(0)]],
                     const device float *b [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant uint &size [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] == b[gid] ? 1.0f : 0.0f;
}

kernel void scalar_eq(const device float *a [[buffer(0)]],
                      device float *out [[buffer(1)]],
                      constant float &val [[buffer(2)]],
                      constant uint &size [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] == val ? 1.0f : 0.0f;
}

kernel void ewise_ge(const device float *a [[buffer(0)]],
                     const device float *b [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant uint &size [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] >= b[gid] ? 1.0f : 0.0f;
}

kernel void scalar_ge(const device float *a [[buffer(0)]],
                      device float *out [[buffer(1)]],
                      constant float &val [[buffer(2)]],
                      constant uint &size [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] >= val ? 1.0f : 0.0f;
}

kernel void ewise_log(const device float *a [[buffer(0)]],
                      device float *out [[buffer(1)]],
                      constant uint &size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = log(a[gid]);
}

kernel void ewise_exp(const device float *a [[buffer(0)]],
                      device float *out [[buffer(1)]],
                      constant uint &size [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = exp(a[gid]);
}

kernel void ewise_tanh(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant uint &size [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = tanh(a[gid]);
}

kernel void ewise_gelu(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant uint &size [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size) {
        float x = a[gid];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}

kernel void matmul_kernel(const device float *a [[buffer(0)]],
                          const device float *b [[buffer(1)]],
                          device float *out [[buffer(2)]],
                          constant uint &M [[buffer(3)]],
                          constant uint &N [[buffer(4)]],
                          constant uint &P [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
    uint row = gid.y;
    uint col = gid.x;
    threadgroup float tile_a[TILE][TILE];
    threadgroup float tile_b[TILE][TILE];

    float acc = 0.0f;
    for (uint base = 0; base < N; base += TILE) {
        uint a_col = base + tid.x;
        uint b_row = base + tid.y;
        tile_a[tid.y][tid.x] =
            (row < M && a_col < N) ? a[row * N + a_col] : 0.0f;
        tile_b[tid.y][tid.x] =
            (b_row < N && col < P) ? b[b_row * P + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k)
            acc += tile_a[tid.y][k] * tile_b[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < P)
        out[row * P + col] = acc;
}

kernel void batched_matmul_kernel(
    const device float *a [[buffer(0)]], const device float *b [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &B [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]],
    constant uint &P [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    uint size = B * M * P;
    if (gid >= size)
        return;
    uint col = gid % P;
    uint row = (gid / P) % M;
    uint batch = gid / (M * P);
    float acc = 0.0f;
    uint a_base = batch * M * N + row * N;
    uint b_base = batch * N * P + col;
    for (uint k = 0; k < N; ++k)
        acc += a[a_base + k] * b[b_base + k * P];
    out[gid] = acc;
}

kernel void reduce_sum(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant uint &reduce_size [[buffer(2)]],
                       constant uint &size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size) {
        float sum = 0.0f;
        for (uint i = 0; i < reduce_size; ++i)
            sum += a[gid * reduce_size + i];
        out[gid] = sum;
    }
}

kernel void reduce_max(const device float *a [[buffer(0)]],
                       device float *out [[buffer(1)]],
                       constant uint &reduce_size [[buffer(2)]],
                       constant uint &size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < size) {
        float maxv = a[gid * reduce_size];
        for (uint i = 1; i < reduce_size; ++i)
            maxv = max(maxv, a[gid * reduce_size + i]);
        out[gid] = maxv;
    }
}

kernel void logsumexp_last_axis_kernel(const device float *a [[buffer(0)]],
                                       device float *out [[buffer(1)]],
                                       constant uint &cols [[buffer(2)]],
                                       constant uint &rows [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (gid >= rows)
        return;
    uint base = gid * cols;
    float maxv = a[base];
    for (uint col = 1; col < cols; ++col)
        maxv = max(maxv, a[base + col]);
    float sum = 0.0f;
    for (uint col = 0; col < cols; ++col)
        sum += exp(a[base + col] - maxv);
    out[gid] = log(sum) + maxv;
}

kernel void softmax_last_axis_kernel(const device float *a [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     constant uint &cols [[buffer(2)]],
                                     constant uint &rows [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= rows)
        return;
    uint base = gid * cols;
    float maxv = a[base];
    for (uint col = 1; col < cols; ++col)
        maxv = max(maxv, a[base + col]);
    float sum = 0.0f;
    for (uint col = 0; col < cols; ++col)
        sum += exp(a[base + col] - maxv);
    for (uint col = 0; col < cols; ++col)
        out[base + col] = exp(a[base + col] - maxv) / sum;
}

kernel void causal_softmax_kernel(const device float *a [[buffer(0)]],
                                  device float *out [[buffer(1)]],
                                  constant uint &seq_len [[buffer(2)]],
                                  constant uint &rows [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= rows)
        return;
    uint query = gid % seq_len;
    uint base = gid * seq_len;
    float maxv = a[base];
    for (uint col = 1; col <= query; ++col)
        maxv = max(maxv, a[base + col]);
    float sum = 0.0f;
    for (uint col = 0; col <= query; ++col)
        sum += exp(a[base + col] - maxv);
    for (uint col = 0; col < seq_len; ++col)
        out[base + col] = col <= query ? exp(a[base + col] - maxv) / sum : 0.0f;
}

kernel void dropout_apply_kernel(const device float *a [[buffer(0)]],
                                 const device float *mask [[buffer(1)]],
                                 device float *out [[buffer(2)]],
                                 constant float &keep_prob [[buffer(3)]],
                                 constant uint &size [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid < size)
        out[gid] = a[gid] * mask[gid] / keep_prob;
}

kernel void embedding_kernel(const device float *weight [[buffer(0)]],
                             const device float *indices [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant uint &count [[buffer(3)]],
                             constant uint &dim [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    uint size = count * dim;
    if (gid >= size)
        return;
    uint token = gid / dim;
    uint feature = gid % dim;
    uint index = uint(indices[token]);
    out[gid] = weight[index * dim + feature];
}

kernel void embedding_backward_kernel(const device float *out_grad
                                      [[buffer(0)]],
                                      const device float *indices [[buffer(1)]],
                                      device atomic_float *weight_grad
                                      [[buffer(2)]],
                                      constant uint &count [[buffer(3)]],
                                      constant uint &dim [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint size = count * dim;
    if (gid >= size)
        return;
    uint token = gid / dim;
    uint feature = gid % dim;
    uint index = uint(indices[token]);
    atomic_fetch_add_explicit(&weight_grad[index * dim + feature],
                              out_grad[gid], memory_order_relaxed);
}

kernel void conv_forward_kernel(
    const device float *x [[buffer(0)]], const device float *w [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant uint &H [[buffer(4)]], constant uint &W [[buffer(5)]],
    constant uint &Cin [[buffer(6)]], constant uint &K [[buffer(7)]],
    constant uint &Cout [[buffer(8)]], constant uint &stride [[buffer(9)]],
    constant uint &padding [[buffer(10)]], constant uint &Hout [[buffer(11)]],
    constant uint &Wout [[buffer(12)]], constant uint &size [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= size)
        return;

    uint co = gid % Cout;
    uint next = gid / Cout;
    uint ow = next % Wout;
    next /= Wout;
    uint oh = next % Hout;
    uint n = next / Hout;

    float acc = 0.0f;
    for (uint kh = 0; kh < K; ++kh) {
        int ih = int(oh * stride + kh) - int(padding);
        if (ih < 0 || ih >= int(H))
            continue;
        for (uint kw = 0; kw < K; ++kw) {
            int iw = int(ow * stride + kw) - int(padding);
            if (iw < 0 || iw >= int(W))
                continue;
            for (uint ci = 0; ci < Cin; ++ci) {
                uint x_idx = ((n * H + uint(ih)) * W + uint(iw)) * Cin + ci;
                uint w_idx = ((kh * K + kw) * Cin + ci) * Cout + co;
                acc += x[x_idx] * w[w_idx];
            }
        }
    }
    out[gid] = acc;
}

kernel void conv_backward_input_kernel(
    const device float *out_grad [[buffer(0)]],
    const device float *w [[buffer(1)]], device float *x_grad [[buffer(2)]],
    constant uint &N [[buffer(3)]], constant uint &H [[buffer(4)]],
    constant uint &W [[buffer(5)]], constant uint &Cin [[buffer(6)]],
    constant uint &K [[buffer(7)]], constant uint &Cout [[buffer(8)]],
    constant uint &stride [[buffer(9)]], constant uint &padding [[buffer(10)]],
    constant uint &Hout [[buffer(11)]], constant uint &Wout [[buffer(12)]],
    constant uint &size [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= size)
        return;

    uint ci = gid % Cin;
    uint next = gid / Cin;
    uint iw = next % W;
    next /= W;
    uint ih = next % H;
    uint n = next / H;

    float acc = 0.0f;
    for (uint kh = 0; kh < K; ++kh) {
        int oh_numer = int(ih) + int(padding) - int(kh);
        if (oh_numer < 0 || oh_numer % int(stride) != 0)
            continue;
        uint oh = uint(oh_numer / int(stride));
        if (oh >= Hout)
            continue;
        for (uint kw = 0; kw < K; ++kw) {
            int ow_numer = int(iw) + int(padding) - int(kw);
            if (ow_numer < 0 || ow_numer % int(stride) != 0)
                continue;
            uint ow = uint(ow_numer / int(stride));
            if (ow >= Wout)
                continue;
            for (uint co = 0; co < Cout; ++co) {
                uint grad_idx = ((n * Hout + oh) * Wout + ow) * Cout + co;
                uint w_idx = ((kh * K + kw) * Cin + ci) * Cout + co;
                acc += out_grad[grad_idx] * w[w_idx];
            }
        }
    }
    x_grad[gid] = acc;
}

kernel void conv_backward_weight_kernel(
    const device float *x [[buffer(0)]],
    const device float *out_grad [[buffer(1)]],
    device float *w_grad [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant uint &H [[buffer(4)]], constant uint &W [[buffer(5)]],
    constant uint &Cin [[buffer(6)]], constant uint &K [[buffer(7)]],
    constant uint &Cout [[buffer(8)]], constant uint &stride [[buffer(9)]],
    constant uint &padding [[buffer(10)]], constant uint &Hout [[buffer(11)]],
    constant uint &Wout [[buffer(12)]], constant uint &size [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= size)
        return;

    uint co = gid % Cout;
    uint next = gid / Cout;
    uint ci = next % Cin;
    next /= Cin;
    uint kw = next % K;
    uint kh = next / K;

    float acc = 0.0f;
    for (uint n = 0; n < N; ++n) {
        for (uint oh = 0; oh < Hout; ++oh) {
            int ih = int(oh * stride + kh) - int(padding);
            if (ih < 0 || ih >= int(H))
                continue;
            for (uint ow = 0; ow < Wout; ++ow) {
                int iw = int(ow * stride + kw) - int(padding);
                if (iw < 0 || iw >= int(W))
                    continue;
                uint x_idx = ((n * H + uint(ih)) * W + uint(iw)) * Cin + ci;
                uint grad_idx = ((n * Hout + oh) * Wout + ow) * Cout + co;
                acc += x[x_idx] * out_grad[grad_idx];
            }
        }
    }
    w_grad[gid] = acc;
}
