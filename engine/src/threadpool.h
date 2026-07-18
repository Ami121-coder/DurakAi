#pragma once

// Простой пул потоков для ISMCTS: N воркеров крутят заданный цикл, пока не
// сработает общий флаг остановки. Использует std::thread (никакого OpenMP),
// что упрощает сборку на MSVC без дополнительных флагов.
//
// Паттерн «rooted parallel MCTS»: одно дерево, защищённое спинлоком; каждый
// воркер независимо крутит determine→select→rollout→backprop. Это чуть хуже
// «tree-parallel» по нагрузке на лок, но проще и хорошо масштабируется до ~12
// потоков на типичных эндшпильно-миттельшпильных деревьях дурака.

#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace durakk {

class ThreadPool {
public:
    // work — функция, вызываемая каждым воркером в цикле до остановки.
    // isTimeUp — атомарный флаг: воркер выходит, когда он становится true.
    void start(int numThreads, std::function<void(int /*workerId*/)> work,
               std::atomic<bool>* stopFlag) {
        stopFlag_ = stopFlag;
        work_ = std::move(work);
        workers_.clear();
        workers_.reserve(numThreads);
        running_ = true;
        for (int i = 0; i < numThreads; ++i) {
            workers_.emplace_back([this, i] { this->work_(i); });
        }
    }

    void join() {
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
        running_ = false;
    }

    ~ThreadPool() { join(); }

private:
    std::vector<std::thread> workers_;
    std::function<void(int)> work_;
    std::atomic<bool>* stopFlag_ = nullptr;
    bool running_ = false;
};

} // namespace durakk
