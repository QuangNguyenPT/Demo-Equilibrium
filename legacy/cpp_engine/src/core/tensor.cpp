#include "hrm/core/tensor.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <iomanip>

namespace hrm {

Tensor::Tensor()
    : shape_({}), strides_({}), dtype_(DataType::Float32), numel_(0), data_(nullptr),
      is_contiguous_(true), is_owner_(false), allocator_(nullptr) {}

Tensor::Tensor(Shape shape, DataType dtype, IAllocator* allocator)
    : shape_(std::move(shape)), dtype_(dtype), is_contiguous_(true), is_owner_(true), allocator_(allocator) {
    numel_ = compute_element_count(shape_);
    strides_ = compute_contiguous_strides(shape_);
    size_t bytes = byte_size();
    if (bytes > 0) {
        if (allocator_) {
            data_ = allocator_->allocate(bytes);
        } else {
            data_ = DefaultAllocator::instance().allocate(bytes);
        }
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(Shape shape, void* external_data, DataType dtype)
    : shape_(std::move(shape)), dtype_(dtype), numel_(compute_element_count(shape_)),
      data_(external_data), is_contiguous_(true), is_owner_(false), allocator_(nullptr) {
    strides_ = compute_contiguous_strides(shape_);
}

Tensor::~Tensor() {
    if (is_owner_ && data_) {
        if (allocator_) {
            allocator_->deallocate(data_);
        } else {
            DefaultAllocator::instance().deallocate(data_);
        }
        data_ = nullptr;
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_),
      numel_(other.numel_), is_contiguous_(other.is_contiguous_), is_owner_(true),
      allocator_(other.allocator_) {
    size_t bytes = byte_size();
    if (bytes > 0) {
        if (allocator_) {
            data_ = allocator_->allocate(bytes);
        } else {
            data_ = DefaultAllocator::instance().allocate(bytes);
        }
        if (other.data_) {
            std::memcpy(data_, other.data_, bytes);
        }
    } else {
        data_ = nullptr;
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    if (is_owner_ && data_) {
        if (allocator_) allocator_->deallocate(data_);
        else DefaultAllocator::instance().deallocate(data_);
        data_ = nullptr;
    }

    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    numel_ = other.numel_;
    is_contiguous_ = other.is_contiguous_;
    allocator_ = other.allocator_;
    is_owner_ = true;

    size_t bytes = byte_size();
    if (bytes > 0) {
        if (allocator_) data_ = allocator_->allocate(bytes);
        else data_ = DefaultAllocator::instance().allocate(bytes);
        if (other.data_) {
            std::memcpy(data_, other.data_, bytes);
        }
    } else {
        data_ = nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      dtype_(other.dtype_), numel_(other.numel_), data_(other.data_),
      is_contiguous_(other.is_contiguous_), is_owner_(other.is_owner_),
      allocator_(other.allocator_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.is_owner_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    if (is_owner_ && data_) {
        if (allocator_) allocator_->deallocate(data_);
        else DefaultAllocator::instance().deallocate(data_);
    }

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    dtype_ = other.dtype_;
    numel_ = other.numel_;
    data_ = other.data_;
    is_contiguous_ = other.is_contiguous_;
    is_owner_ = other.is_owner_;
    allocator_ = other.allocator_;

    other.data_ = nullptr;
    other.numel_ = 0;
    other.is_owner_ = false;
    return *this;
}

void Tensor::zero() {
    if (data_ && byte_size() > 0) {
        std::memset(data_, 0, byte_size());
    }
}

void Tensor::fill(float value) {
    if (!data_ || dtype_ != DataType::Float32) return;
    float* ptr = data<float>();
    for (int64_t i = 0; i < numel_; ++i) {
        ptr[i] = value;
    }
}

void Tensor::copy_from(const void* src, size_t bytes) {
    if (!data_ || !src) return;
    size_t copy_bytes = (bytes < byte_size()) ? bytes : byte_size();
    std::memcpy(data_, src, copy_bytes);
}

void Tensor::copy_to(void* dst, size_t bytes) const {
    if (!data_ || !dst) return;
    size_t copy_bytes = (bytes < byte_size()) ? bytes : byte_size();
    std::memcpy(dst, data_, copy_bytes);
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    int64_t new_numel = compute_element_count(new_shape);
    if (new_numel != numel_) {
        throw std::invalid_argument("[TENSOR] Cannot reshape: element count mismatch.");
    }
    Tensor t = *this;
    t.shape_ = new_shape;
    t.strides_ = compute_contiguous_strides(new_shape);
    return t;
}

TensorView Tensor::view() const {
    return TensorView(*this, shape_, 0);
}

void Tensor::print_summary(const std::string& tag) const {
    std::cout << "[" << tag << "] Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    std::cout << "], DType: " << data_type_name(dtype_)
              << ", Numel: " << numel_
              << ", Bytes: " << byte_size()
              << ", Contiguous: " << (is_contiguous_ ? "true" : "false") << "\n";
}

TensorView::TensorView(const Tensor& tensor, Shape slice_shape, int64_t byte_offset)
    : shape_(std::move(slice_shape)), dtype_(tensor.dtype()) {
    const uint8_t* raw = static_cast<const uint8_t*>(tensor.raw_data());
    data_ = raw ? (raw + byte_offset) : nullptr;
}

} // namespace hrm
