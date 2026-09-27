#include "hrm/runtime/thread_pool.hpp"

namespace hrm {
namespace runtime {

ThreadPool::ThreadPool(size_t threads) {
    if (threads == 0) threads = 1;
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    this->cv_.wait(lock, [this]() {
                        return this->stop_ || !this->tasks_.empty();
                    });
                    if (this->stop_ && this->tasks_.empty()) {
                        return;
                    }
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::parallel_for(int64_t begin, int64_t end, const std::function<void(int64_t)>& loop_body) {
    if (begin >= end) return;
    int64_t total = end - begin;
    size_t num_threads = workers_.size();
    if (num_threads <= 1 || total <= 1) {
        for (int64_t i = begin; i < end; ++i) {
            loop_body(i);
        }
        return;
    }

    int64_t chunk_size = (total + static_cast<int64_t>(num_threads) - 1) / static_cast<int64_t>(num_threads);
    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
        int64_t c_start = begin + static_cast<int64_t>(t) * chunk_size;
        int64_t c_end = std::min(c_start + chunk_size, end);
        if (c_start >= end) break;

        futures.push_back(enqueue([c_start, c_end, &loop_body]() {
            for (int64_t i = c_start; i < c_end; ++i) {
                loop_body(i);
            }
        }));
    }

    for (auto& fut : futures) {
        fut.get();
    }
}

} // namespace runtime
} // namespace hrm
