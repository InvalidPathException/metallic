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