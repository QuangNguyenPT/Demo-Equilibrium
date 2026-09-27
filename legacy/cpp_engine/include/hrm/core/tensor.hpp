#pragma once

#include "hrm/core/types.hpp"
#include "hrm/core/allocator.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace hrm {

class TensorView;

class Tensor {
public:
    Tensor();
    Tensor(Shape shape, DataType dtype = DataType::Float32, IAllocator* allocator = nullptr);
    Tensor(Shape shape, void* external_data, DataType dtype = DataType::Float32); // Non-owning wrap
    ~Tensor();

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Zero-fill / Constant fill
    void zero();
    void fill(float value);
    void copy_from(const void* src, size_t bytes);
    void copy_to(void* dst, size_t bytes) const;

    // Metadata
    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    DataType dtype() const { return dtype_; }
    int64_t numel() const { return numel_; }
    size_t byte_size() const { return numel_ * data_type_size(dtype_); }
    bool is_contiguous() const { return is_contiguous_; }
    bool is_owner() const { return is_owner_; }

    // Pointer access
    void* raw_data() { return data_; }
    const void* raw_data() const { return data_; }

    template <typename T>
    T* data() { return static_cast<T*>(data_); }

    template <typename T>
    const T* data() const { return static_cast<const T*>(data_); }

    // Reshape (returns new view or reconfigured shape if element count identical)
    Tensor reshape(const Shape& new_shape) const;
    TensorView view() const;

    void print_summary(const std::string& tag = "Tensor") const;

private:
    Shape shape_;
    Strides strides_;
    DataType dtype_{DataType::Float32};
    int64_t numel_{0};
    void* data_{nullptr};
    bool is_contiguous_{true};
    bool is_owner_{false};
    IAllocator* allocator_{nullptr};
};

class TensorView {
public:
    TensorView(const Tensor& tensor, Shape slice_shape, int64_t byte_offset);

    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    const void* raw_data() const { return data_; }

private:
    Shape shape_;
    DataType dtype_;
    const void* data_{nullptr};
};

} // namespace hrm
