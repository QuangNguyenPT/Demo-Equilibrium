#include "hrm/core/allocator.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

namespace hrm {

void* aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

DefaultAllocator& DefaultAllocator::instance() {
    static DefaultAllocator s_instance;
    return s_instance;
}

void* DefaultAllocator::allocate(size_t bytes, size_t alignment) {
    void* ptr = aligned_alloc(bytes, alignment);
    if (!ptr && bytes > 0) {
        throw std::bad_alloc();
    }
    return ptr;
}

void DefaultAllocator::deallocate(void* ptr) {
    aligned_free(ptr);
}

ArenaAllocator::ArenaAllocator(size_t capacity_bytes, size_t alignment)
    : capacity_(capacity_bytes), offset_(0), default_alignment_(alignment) {
    buffer_ = static_cast<uint8_t*>(aligned_alloc(capacity_bytes, default_alignment_));
    if (!buffer_ && capacity_bytes > 0) {
        throw std::bad_alloc();
    }
}

ArenaAllocator::~ArenaAllocator() {
    if (buffer_) {
        aligned_free(buffer_);
        buffer_ = nullptr;
    }
}

void* ArenaAllocator::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    std::lock_guard<std::mutex> lock(mutex_);

    // Align current offset
    size_t align_mask = alignment - 1;
    size_t current_addr = reinterpret_cast<size_t>(buffer_ + offset_);
    size_t padding = (alignment - (current_addr & align_mask)) & align_mask;

    size_t next_offset = offset_ + padding + bytes;
    if (next_offset > capacity_) {
        throw std::runtime_error("[ARENA] Out of memory in ArenaAllocator: requested " +
                                 std::to_string(bytes) + " bytes, capacity " +
                                 std::to_string(capacity_) + " bytes.");
    }

    void* allocated_ptr = buffer_ + offset_ + padding;
    offset_ = next_offset;
    return allocated_ptr;
}

void ArenaAllocator::deallocate(void* /*ptr*/) {
    // Zero-overhead bump allocator: no-op per object, freed entirely via reset()
}

void ArenaAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    offset_ = 0;
}

size_t ArenaAllocator::used_bytes() const {
    return offset_;
}

size_t ArenaAllocator::total_capacity() const {
    return capacity_;
}

} // namespace hrm
