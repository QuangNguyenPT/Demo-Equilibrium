#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

namespace hrm {

enum class DataType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int8 = 4
};

inline size_t data_type_size(DataType dt) {
    switch (dt) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int8: return 1;
        default: return 4;
    }
}

inline const char* data_type_name(DataType dt) {
    switch (dt) {
        case DataType::Float32: return "Float32";
        case DataType::Float16: return "Float16";
        case DataType::BFloat16: return "BFloat16";
        case DataType::Int32: return "Int32";
        case DataType::Int8: return "Int8";
        default: return "Unknown";
    }
}

enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1
};

enum class StatusCode : int32_t {
    Success = 0,
    InvalidDimensions = 1,
    OutOfMemory = 2,
    NullPointer = 3,
    NotImplemented = 4,
    ComputationError = 5
};

using Shape = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

inline int64_t compute_element_count(const Shape& shape) {
    if (shape.empty()) return 0;
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

inline Strides compute_contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    if (shape.empty()) return strides;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

} // namespace hrm
