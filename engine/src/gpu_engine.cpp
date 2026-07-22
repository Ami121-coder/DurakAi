#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// CUDA launcher declarations (из gpu_launchers.cu)
extern "C" {
    void launchRollouts(const GpuGameState* states, int* results, int N,
                        int botPlayer, unsigned long long seed, void* stream);
    void launchEval(const GpuGameState* states, float* scores, int N,
                    int botPlayer, void* stream);
    void launchSample(uint64_t* sampled, uint64_t known, const float* weights,
                      int handSize, int N, unsigned long long seed, void* stream);
    void launchReduce(const int* results, int* wins, int* losses, int* draws,
                      int N, void* stream);
    void uploadEvalParams(const EvalParams& params);
}

static constexpr size_t STATE_SIZE = 64;
static constexpr size_t DEFAULT_CAPACITY = 524288; // 512K состояний = 32 MB

GpuEngine::GpuEngine() {}
GpuEngine::~GpuEngine() { shutdown(); }

bool GpuEngine::init(int deviceIndex) {
    cudaError_t err = cudaSetDevice(deviceIndex);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU] cudaSetDevice: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceIndex);
    printf("[GPU] %s | SM %d.%d | %d SMs | %.1f GB VRAM\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    capacity_ = DEFAULT_CAPACITY;

    // Преаллокация ВСЕХ буферов (ФИКС #5: никаких malloc в hot path)
    for (int b = 0; b < 2; ++b) {
        cudaMalloc(&d_states_[b], capacity_ * STATE_SIZE);
        cudaMalloc(&d_results_[b], capacity_ * sizeof(int));
    }
    cudaMalloc(&d_scores_, capacity_ * sizeof(float));
    cudaMalloc(&d_sampled_, capacity_ * sizeof(uint64_t));
    cudaMalloc(&d_weights_, 36 * sizeof(float));
    cudaMalloc(&d_wins_, sizeof(int));
    cudaMalloc(&d_losses_, sizeof(int));
    cudaMalloc(&d_draws_, sizeof(int));

    // CUDA streams для async overlap (ФИКС #6)
    for (int s = 0; s < 2; ++s) {
        cudaStreamCreate((cudaStream_t*)&streams_[s]);
    }

    ready_ = true;
    printf("[GPU] Buffers: %zuK states (%.1f MB), 2 streams, preallocated\n",
           capacity_ / 1024, capacity_ * STATE_SIZE / 1e6);
    return true;
}

void GpuEngine::shutdown() {
    for (int b = 0; b < 2; ++b) {
        if (d_states_[b]) cudaFree(d_states_[b]);
        if (d_results_[b]) cudaFree(d_results_[b]);
        if (streams_[b]) cudaStreamDestroy((cudaStream_t)streams_[b]);
    }
    if (d_scores_) cudaFree(d_scores_);
    if (d_sampled_) cudaFree(d_sampled_);
    if (d_weights_) cudaFree(d_weights_);
    if (d_wins_) cudaFree(d_wins_);
    if (d_losses_) cudaFree(d_losses_);
    if (d_draws_) cudaFree(d_draws_);
    memset(this, 0, sizeof(*this)); // обнуляем указатели
}

void GpuEngine::ensureCapacity(size_t n) {
    if (n <= capacity_) return;
    // Реаллокация (редко, только если батч > 512K)
    size_t newCap = n;
    for (int b = 0; b < 2; ++b) {
        cudaFree(d_states_[b]);
        cudaFree(d_results_[b]);
        cudaMalloc(&d_states_[b], newCap * STATE_SIZE);
        cudaMalloc(&d_results_[b], newCap * sizeof(int));
    }
    cudaFree(d_scores_);
    cudaMalloc(&d_scores_, newCap * sizeof(float));
    cudaFree(d_sampled_);
    cudaMalloc(&d_sampled_, newCap * sizeof(uint64_t));
    capacity_ = newCap;
    printf("[GPU] Reallocated to %zuK states\n", newCap / 1024);
}

GpuRolloutResult GpuEngine::runRollouts(
    const std::vector<GpuGameState>& states, int botPlayer
) {
    GpuRolloutResult res;
    if (!ready_ || states.empty()) return res;

    int N = (int)states.size();
    ensureCapacity(N);
    int buf = 0;

    // Upload
    cudaMemcpyAsync(d_states_[buf], states.data(), N * STATE_SIZE,
                    cudaMemcpyHostToDevice, (cudaStream_t)streams_[buf]);

    // Reset counters
    cudaMemsetAsync(d_wins_, 0, sizeof(int), (cudaStream_t)streams_[buf]);
    cudaMemsetAsync(d_losses_, 0, sizeof(int), (cudaStream_t)streams_[buf]);
    cudaMemsetAsync(d_draws_, 0, sizeof(int), (cudaStream_t)streams_[buf]);

    // Launch rollouts
    launchRollouts((const GpuGameState*)d_states_[buf],
                   (int*)d_results_[buf], N, botPlayer, rngSeed_,
                   streams_[buf]);
    rngSeed_ += 7919;

    // Reduce
    launchReduce((const int*)d_results_[buf],
                 (int*)d_wins_, (int*)d_losses_, (int*)d_draws_,
                 N, streams_[buf]);

    // Sync + download
    cudaStreamSynchronize((cudaStream_t)streams_[buf]);
    cudaMemcpy(&res.wins, d_wins_, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.losses, d_losses_, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.draws, d_draws_, sizeof(int), cudaMemcpyDeviceToHost);

    int total = res.wins + res.losses + res.draws;
    res.winRate = (total > 0) ? (float)(res.wins + 0.5f * res.draws) / total : 0.5f;
    return res;
}

void GpuEngine::runRolloutsAsync(
    const std::vector<GpuGameState>& states, int botPlayer,
    std::function<void(GpuRolloutResult)> callback
) {
    if (!ready_ || states.empty()) { callback({}); return; }

    int N = (int)states.size();
    ensureCapacity(N);
    int buf = curBuf_;
    curBuf_ = 1 - curBuf_; // flip

    cudaMemcpyAsync(d_states_[buf], states.data(), N * STATE_SIZE,
                    cudaMemcpyHostToDevice, (cudaStream_t)streams_[buf]);
    cudaMemsetAsync(d_wins_, 0, sizeof(int), (cudaStream_t)streams_[buf]);
    cudaMemsetAsync(d_losses_, 0, sizeof(int), (cudaStream_t)streams_[buf]);
    cudaMemsetAsync(d_draws_, 0, sizeof(int), (cudaStream_t)streams_[buf]);

    launchRollouts((const GpuGameState*)d_states_[buf],
                   (int*)d_results_[buf], N, botPlayer, rngSeed_,
                   streams_[buf]);
    rngSeed_ += 7919;

    launchReduce((const int*)d_results_[buf],
                 (int*)d_wins_, (int*)d_losses_, (int*)d_draws_,
                 N, streams_[buf]);

    asyncPending_ = true;
    // Callback вызывается в pollAsync()
}

bool GpuEngine::pollAsync(GpuRolloutResult& out) {
    if (!asyncPending_) return false;
    int buf = 1 - curBuf_; // предыдущий буфер
    cudaError_t status = cudaStreamQuery((cudaStream_t)streams_[buf]);
    if (status == cudaSuccess) {
        cudaMemcpy(&out.wins, d_wins_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&out.losses, d_losses_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&out.draws, d_draws_, sizeof(int), cudaMemcpyDeviceToHost);
        int total = out.wins + out.losses + out.draws;
        out.winRate = (total > 0) ? (float)(out.wins + 0.5f * out.draws) / total : 0.5f;
        asyncPending_ = false;
        return true;
    }
    return false; // ещё не готово
}

std::vector<float> GpuEngine::evaluateBatch(
    const std::vector<GpuGameState>& states, int botPlayer
) {
    std::vector<float> scores(states.size(), 0.0f);
    if (!ready_ || states.empty()) return scores;

    int N = (int)states.size();
    ensureCapacity(N);
    cudaMemcpyAsync(d_states_[0], states.data(), N * STATE_SIZE,
                    cudaMemcpyHostToDevice, (cudaStream_t)streams_[0]);
    launchEval((const GpuGameState*)d_states_[0], (float*)d_scores_,
               N, botPlayer, streams_[0]);
    cudaStreamSynchronize((cudaStream_t)streams_[0]);
    cudaMemcpy(scores.data(), d_scores_, N * sizeof(float), cudaMemcpyDeviceToHost);
    return scores;
}

std::vector<uint64_t> GpuEngine::sampleOpponentHands(
    uint64_t knownCards, const float weights[36], int handSize, int count
) {
    std::vector<uint64_t> hands(count, 0);
    if (!ready_ || count <= 0) return hands;

    int N = std::min(count, (int)capacity_);
    cudaMemcpyAsync(d_weights_, weights, 36 * sizeof(float),
                    cudaMemcpyHostToDevice, (cudaStream_t)streams_[0]);
    launchSample((uint64_t*)d_sampled_, knownCards, (const float*)d_weights_,
                 handSize, N, rngSeed_, streams_[0]);
    rngSeed_ += 104729;
    cudaStreamSynchronize((cudaStream_t)streams_[0]);
    cudaMemcpy(hands.data(), d_sampled_, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return hands;
}

void GpuEngine::setEvalParams(const EvalParams& params) {
    if (ready_) uploadEvalParams(params);
}

GpuEngine::DeviceInfo GpuEngine::getDeviceInfo() const {
    DeviceInfo info{};
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    strncpy(info.name, prop.name, 255);
    info.smCount = prop.multiProcessorCount;
    info.totalMem = prop.totalGlobalMem;
    return info;
}