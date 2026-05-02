#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal_kernels.hpp"

namespace metallic {

namespace {

constexpr size_t kBaseThreadCount = 256;
constexpr size_t kTileSize = 8;
constexpr size_t kMaxVecSize = 8;

using Scalar = float;
constexpr size_t kElemSize = sizeof(Scalar);

struct MetalVec {
    uint32_t size;
    int32_t data[kMaxVecSize];
};

uint32_t U32(size_t value) {
    if (value > UINT32_MAX) {
        throw std::runtime_error(
            "Metal backend only supports 32-bit element counts");
    }
    return static_cast<uint32_t>(value);
}

MetalVec ToMetalVec(const std::vector<int32_t> &values) {
    if (values.size() > kMaxVecSize) {
        throw std::runtime_error("Exceeded Metal supported max dimensions");
    }

    MetalVec out{};
    out.size = U32(values.size());
    std::copy(values.begin(), values.end(), out.data);
    return out;
}

class Runtime {
  public:
    static Runtime &Get() {
        static Runtime runtime;
        return runtime;
    }

    MTL::Device &Device() { return *device_; }

    template <typename Bind>
    void Launch1D(const char *kernel, size_t size, Bind &&bind) {
        auto *pipeline = Pipeline(kernel);
        auto *command_buffer = queue_->commandBuffer();
        auto *encoder = command_buffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline);
        bind(*encoder);
        encoder->dispatchThreads(
            MTL::Size(size, 1, 1),
            MTL::Size(std::min(kBaseThreadCount,
                               pipeline->maxTotalThreadsPerThreadgroup()),
                      1, 1));
        encoder->endEncoding();

        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        Check(*command_buffer);
    }

    template <typename Bind>
    void Launch2DThreadgroups(const char *kernel, size_t width, size_t height,
                              size_t group_width, size_t group_height,
                              Bind &&bind) {
        auto *pipeline = Pipeline(kernel);
        auto *command_buffer = queue_->commandBuffer();
        auto *encoder = command_buffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline);
        bind(*encoder);
        encoder->dispatchThreadgroups(
            MTL::Size((width + group_width - 1) / group_width,
                      (height + group_height - 1) / group_height, 1),
            MTL::Size(group_width, group_height, 1));
        encoder->endEncoding();

        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        Check(*command_buffer);
    }

  private:
    Runtime() {
        pool_ = NS::AutoreleasePool::alloc()->init();
        device_ = MTL::CreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("No Metal device available");
        }

        auto *source =
            NS::String::string(kMetalKernels, NS::UTF8StringEncoding);
        NS::Error *error = nullptr;
        library_ = device_->newLibrary(source, nullptr, &error);
        if (!library_) {
            std::string message =
                error ? error->localizedDescription()->utf8String()
                      : "unknown error";
            throw std::runtime_error("Failed to compile Metal kernels: " +
                                     message);
        }

        queue_ = device_->newCommandQueue();
        if (!queue_) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
    }

    ~Runtime() {
        for (auto &[_, pipeline] : pipelines_) {
            pipeline->release();
        }
        queue_->release();
        library_->release();
        device_->release();
        pool_->release();
    }

    MTL::ComputePipelineState *Pipeline(const char *name) {
        auto it = pipelines_.find(name);
        if (it != pipelines_.end()) {
            return it->second;
        }

        auto *fn_name = NS::String::string(name, NS::UTF8StringEncoding);
        auto *fn = library_->newFunction(fn_name);
        if (!fn) {
            throw std::runtime_error(std::string("Metal function not found: ") +
                                     name);
        }

        NS::Error *error = nullptr;
        auto *pipeline = device_->newComputePipelineState(fn, &error);
        fn->release();
        if (!pipeline) {
            std::string message =
                error ? error->localizedDescription()->utf8String()
                      : "unknown error";
            throw std::runtime_error("Failed to create Metal pipeline " +
                                     std::string(name) + ": " + message);
        }

        pipelines_[name] = pipeline;
        return pipeline;
    }

    void Check(const MTL::CommandBuffer &command_buffer) {
        if (command_buffer.status() != MTL::CommandBufferStatusError) {
            return;
        }

        auto *error = command_buffer.error();
        std::string message = error
                                  ? error->localizedDescription()->utf8String()
                                  : "unknown error";
        throw std::runtime_error("Metal command buffer failed: " + message);
    }

    NS::AutoreleasePool *pool_ = nullptr;
    MTL::Device *device_ = nullptr;
    MTL::Library *library_ = nullptr;
    MTL::CommandQueue *queue_ = nullptr;
    std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines_;
};

struct Array {
    explicit Array(size_t size) : size(size) {
        buffer = Runtime::Get().Device().newBuffer(
            size * kElemSize, MTL::ResourceStorageModeShared);
        if (!buffer) {
            throw std::bad_alloc();
        }
    }

    ~Array() { buffer->release(); }

    size_t ptr_as_int() const { return reinterpret_cast<size_t>(buffer); }

    size_t size;
    MTL::Buffer *buffer;
};

void SetBuffer(MTL::ComputeCommandEncoder &encoder, uint32_t slot,
               const Array &array) {
    encoder.setBuffer(array.buffer, 0, slot);
}

template <typename T>
void SetValue(MTL::ComputeCommandEncoder &encoder, uint32_t slot,
              const T &value) {
    encoder.setBytes(&value, sizeof(T), slot);
}

void Fill(Array *out, Scalar value) {
    uint32_t size = U32(out->size);
    Runtime::Get().Launch1D("fill_kernel", out->size, [&](auto &encoder) {
        SetBuffer(encoder, 0, *out);
        SetValue(encoder, 1, value);
        SetValue(encoder, 2, size);
    });
}

template <const char *Kernel, typename SizeSource>
void StridedWrite(const Array &a, Array *out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset,
                  SizeSource source_size) {
    uint32_t size = U32(source_size());
    uint32_t off = U32(offset);
    MetalVec shape_vec = ToMetalVec(shape);
    MetalVec stride_vec = ToMetalVec(strides);

    Runtime::Get().Launch1D(Kernel, source_size(), [&](auto &encoder) {
        SetBuffer(encoder, 0, a);
        SetBuffer(encoder, 1, *out);
        SetValue(encoder, 2, size);
        SetValue(encoder, 3, shape_vec);
        SetValue(encoder, 4, stride_vec);
        SetValue(encoder, 5, off);
    });
}

void ScalarSetitem(size_t item_count, Scalar value, Array *out,
                   std::vector<int32_t> shape, std::vector<int32_t> strides,
                   size_t offset) {
    uint32_t size = U32(item_count);
    uint32_t off = U32(offset);
    MetalVec shape_vec = ToMetalVec(shape);
    MetalVec stride_vec = ToMetalVec(strides);

    Runtime::Get().Launch1D("scalar_setitem_kernel", item_count,
                            [&](auto &encoder) {
                                SetBuffer(encoder, 0, *out);
                                SetValue(encoder, 1, value);
                                SetValue(encoder, 2, size);
                                SetValue(encoder, 3, shape_vec);
                                SetValue(encoder, 4, stride_vec);
                                SetValue(encoder, 5, off);
                            });
}

void BinaryOp(const char *kernel, const Array &a, const Array &b, Array *out) {
    uint32_t size = U32(out->size);
    Runtime::Get().Launch1D(kernel, out->size, [&](auto &encoder) {
        SetBuffer(encoder, 0, a);
        SetBuffer(encoder, 1, b);
        SetBuffer(encoder, 2, *out);
        SetValue(encoder, 3, size);
    });
}

void ScalarOp(const char *kernel, const Array &a, Scalar value, Array *out) {
    uint32_t size = U32(out->size);
    Runtime::Get().Launch1D(kernel, out->size, [&](auto &encoder) {
        SetBuffer(encoder, 0, a);
        SetBuffer(encoder, 1, *out);
        SetValue(encoder, 2, value);
        SetValue(encoder, 3, size);
    });
}

void UnaryOp(const char *kernel, const Array &a, Array *out) {
    uint32_t size = U32(out->size);
    Runtime::Get().Launch1D(kernel, out->size, [&](auto &encoder) {
        SetBuffer(encoder, 0, a);
        SetBuffer(encoder, 1, *out);
        SetValue(encoder, 2, size);
    });
}

void ReduceOp(const char *kernel, const Array &a, Array *out,
              size_t reduce_size) {
    uint32_t reduce_size_u32 = U32(reduce_size);
    uint32_t size = U32(out->size);
    Runtime::Get().Launch1D(kernel, out->size, [&](auto &encoder) {
        SetBuffer(encoder, 0, a);
        SetBuffer(encoder, 1, *out);
        SetValue(encoder, 2, reduce_size_u32);
        SetValue(encoder, 3, size);
    });
}

void Matmul(const Array &a, const Array &b, Array *out, uint32_t rows,
            uint32_t inner, uint32_t cols) {
    Runtime::Get().Launch2DThreadgroups("matmul_kernel", cols, rows, kTileSize,
                                        kTileSize, [&](auto &encoder) {
                                            SetBuffer(encoder, 0, a);
                                            SetBuffer(encoder, 1, b);
                                            SetBuffer(encoder, 2, *out);
                                            SetValue(encoder, 3, rows);
                                            SetValue(encoder, 4, inner);
                                            SetValue(encoder, 5, cols);
                                        });
}

void ConvForward(const Array &x, const Array &w, Array *out, uint32_t n,
                 uint32_t h, uint32_t width, uint32_t cin, uint32_t kernel,
                 uint32_t cout, uint32_t stride, uint32_t padding,
                 uint32_t out_h, uint32_t out_w) {
    uint32_t size = U32(out->size);
    Runtime::Get().Launch1D("conv_forward_kernel", out->size,
                            [&](auto &encoder) {
                                SetBuffer(encoder, 0, x);
                                SetBuffer(encoder, 1, w);
                                SetBuffer(encoder, 2, *out);
                                SetValue(encoder, 3, n);
                                SetValue(encoder, 4, h);
                                SetValue(encoder, 5, width);
                                SetValue(encoder, 6, cin);
                                SetValue(encoder, 7, kernel);
                                SetValue(encoder, 8, cout);
                                SetValue(encoder, 9, stride);
                                SetValue(encoder, 10, padding);
                                SetValue(encoder, 11, out_h);
                                SetValue(encoder, 12, out_w);
                                SetValue(encoder, 13, size);
                            });
}

void ConvBackwardInput(const Array &out_grad, const Array &w, Array *x_grad,
                       uint32_t n, uint32_t h, uint32_t width, uint32_t cin,
                       uint32_t kernel, uint32_t cout, uint32_t stride,
                       uint32_t padding, uint32_t out_h, uint32_t out_w) {
    uint32_t size = U32(x_grad->size);
    Runtime::Get().Launch1D("conv_backward_input_kernel", x_grad->size,
                            [&](auto &encoder) {
                                SetBuffer(encoder, 0, out_grad);
                                SetBuffer(encoder, 1, w);
                                SetBuffer(encoder, 2, *x_grad);
                                SetValue(encoder, 3, n);
                                SetValue(encoder, 4, h);
                                SetValue(encoder, 5, width);
                                SetValue(encoder, 6, cin);
                                SetValue(encoder, 7, kernel);
                                SetValue(encoder, 8, cout);
                                SetValue(encoder, 9, stride);
                                SetValue(encoder, 10, padding);
                                SetValue(encoder, 11, out_h);
                                SetValue(encoder, 12, out_w);
                                SetValue(encoder, 13, size);
                            });
}

void ConvBackwardWeight(const Array &x, const Array &out_grad, Array *w_grad,
                        uint32_t n, uint32_t h, uint32_t width, uint32_t cin,
                        uint32_t kernel, uint32_t cout, uint32_t stride,
                        uint32_t padding, uint32_t out_h, uint32_t out_w) {
    uint32_t size = U32(w_grad->size);
    Runtime::Get().Launch1D("conv_backward_weight_kernel", w_grad->size,
                            [&](auto &encoder) {
                                SetBuffer(encoder, 0, x);
                                SetBuffer(encoder, 1, out_grad);
                                SetBuffer(encoder, 2, *w_grad);
                                SetValue(encoder, 3, n);
                                SetValue(encoder, 4, h);
                                SetValue(encoder, 5, width);
                                SetValue(encoder, 6, cin);
                                SetValue(encoder, 7, kernel);
                                SetValue(encoder, 8, cout);
                                SetValue(encoder, 9, stride);
                                SetValue(encoder, 10, padding);
                                SetValue(encoder, 11, out_h);
                                SetValue(encoder, 12, out_w);
                                SetValue(encoder, 13, size);
                            });
}

constexpr const char *kBinaryOps[] = {
    "ewise_add",     "ewise_mul", "ewise_div",
    "ewise_maximum", "ewise_eq",  "ewise_ge",
};

constexpr const char *kScalarOps[] = {
    "scalar_add",     "scalar_mul", "scalar_div", "scalar_power",
    "scalar_maximum", "scalar_eq",  "scalar_ge",
};

constexpr const char *kUnaryOps[] = {
    "ewise_log",
    "ewise_exp",
    "ewise_tanh",
};

constexpr const char *kReduceOps[] = {
    "reduce_max",
    "reduce_sum",
};

constexpr char kCompactKernel[] = "compact_kernel";
constexpr char kEwiseSetitemKernel[] = "ewise_setitem_kernel";

} // namespace

} // namespace metallic

PYBIND11_MODULE(ndarray_backend, m) {
    namespace py = pybind11;
    using namespace metallic;

    m.attr("__device_name__") = "metal";
    m.attr("__tile_size__") = kTileSize;

    py::class_<Array>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &Array::size)
        .def("ptr", &Array::ptr_as_int);

    m.def("to_numpy", [](const Array &a, std::vector<size_t> shape,
                         std::vector<size_t> strides, size_t offset) {
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(),
                       numpy_strides.begin(),
                       [](size_t stride) { return stride * kElemSize; });

        auto *host_ptr = static_cast<Scalar *>(std::malloc(a.size * kElemSize));
        if (!host_ptr) {
            throw std::bad_alloc();
        }
        std::memcpy(host_ptr, a.buffer->contents(), a.size * kElemSize);

        py::capsule deallocate_buffer(host_ptr,
                                      [](void *ptr) { std::free(ptr); });
        return py::array_t<Scalar>(shape, numpy_strides, host_ptr + offset,
                                   deallocate_buffer);
    });

    m.def("from_numpy", [](py::array_t<Scalar> a, Array *out) {
        auto req = a.request();
        std::memcpy(out->buffer->contents(), req.ptr, out->size * kElemSize);
    });

    m.def("fill", Fill);
    m.def("compact", [](const Array &a, Array *out, std::vector<int32_t> shape,
                        std::vector<int32_t> strides, size_t offset) {
        StridedWrite<kCompactKernel>(a, out, std::move(shape),
                                     std::move(strides), offset,
                                     [&] { return out->size; });
    });
    m.def("ewise_setitem",
          [](const Array &a, Array *out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
              StridedWrite<kEwiseSetitemKernel>(a, out, std::move(shape),
                                                std::move(strides), offset,
                                                [&] { return a.size; });
          });
    m.def("scalar_setitem", ScalarSetitem);

    for (const char *kernel : kBinaryOps) {
        m.def(kernel, [kernel](const Array &a, const Array &b, Array *out) {
            BinaryOp(kernel, a, b, out);
        });
    }

    for (const char *kernel : kScalarOps) {
        m.def(kernel, [kernel](const Array &a, Scalar value, Array *out) {
            ScalarOp(kernel, a, value, out);
        });
    }

    for (const char *kernel : kUnaryOps) {
        m.def(kernel, [kernel](const Array &a, Array *out) {
            UnaryOp(kernel, a, out);
        });
    }

    for (const char *kernel : kReduceOps) {
        m.def(kernel, [kernel](const Array &a, Array *out, size_t reduce_size) {
            ReduceOp(kernel, a, out, reduce_size);
        });
    }

    m.def("matmul", Matmul);
    m.def("conv_forward", ConvForward);
    m.def("conv_backward_input", ConvBackwardInput);
    m.def("conv_backward_weight", ConvBackwardWeight);
}
