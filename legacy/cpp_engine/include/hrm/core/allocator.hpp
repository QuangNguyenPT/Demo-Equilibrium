#pragma once

#include "hrm/core/types.hpp"
#include <memory>
#include <mutex>

namespace hrm {

constexpr size_t DEFAULT_ALIGNMENT = 64; // Cache-line alignment for AVX-512 / AVX2

void* aligned_alloc(size_t size, size_t alignment = DEFAULT_ALIGNMENT);
void aligned_free(void* ptr);

class IAllocator {
public:
    virtual ~IAllocator() = default;
    virtual void* allocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) = 0;
    virtual void deallocate(void* ptr) = 0;
};

class DefaultAllocator : public IAllocator {
public:
    static DefaultAllocator& instance();
    void* allocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) override;
    void deallocate(void* ptr) override;
};

// Zero-fragmentation bump arena for fast forward inference passes
class ArenaAllocator : public IAllocator {
public:
    explicit ArenaAllocator(size_t capacity_bytes, size_t alignment = DEFAULT_ALIGNMENT);
    ~ArenaAllocator() override;

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    void* allocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) override;
    void deallocate(void* ptr) override; // No-op in bump arena

    void reset(); // Instant zero-cost reset between model passes
    size_t used_bytes() const;
    size_t total_capacity() const;

private:
    uint8_t* buffer_{nullptr};
    size_t capacity_{0};
    size_t offset_{0};
    size_t default_alignment_{DEFAULT_ALIGNMENT};
    std::mutex mutex_;
};

} // namespace hrm
