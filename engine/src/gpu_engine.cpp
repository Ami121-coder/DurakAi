#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// Объявления CUDA-ядер (из gpu_kernels.cu)
extern "C" {
    void launchRollouts(const GpuGameState* states, int* results, int N, int botPlayer, unsigned long long seed);
    void launchEval(const GpuGameState* states, float* scores, int N, int botPlayer);
    void launchSample(uint64_t* sampled, uint64_t known, const float* weights, int handSize, int N, unsigned long long seed);
    void launchReduce(const int* results, int* wins, int* losses, int* draws, int N);
    void uploadEvalParams(const EvalParams& params);
}

// Размер состояния (должен совпадать с CUDA struct)
static constexpr size_t GPU_STATE_SIZE = 64;

GpuEngine::GpuEngine() {}

GpuEngine::~GpuEngine() { shutdown(); }

bool GpuEngine::init(int deviceIndex) {
    cudaError_t err;

    err = cudaSetDevice(deviceIndex);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU] cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Проверяем устройство
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceIndex);
    printf("[GPU] Device: %s, SM %d.%d, %d SMs, %.1f GB VRAM\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // RTX 4060 Ti: SM 8.9, 34 SM, 8 GB
    // Выделяем буферы под 1M состояний (64 MB) — комфортно для 8 GB
    allocStates_ = 1024 * 1024; // 1M
    allocResults_ = allocStates_;

    err = cudaMalloc(&d_states_, allocStates_ * GPU_STATE_SIZE);
    if (err != cudaSuccess) { fprintf(stderr, "[GPU] alloc states failed\n"); return false; }

    err = cudaMalloc(&d_results_, allocResults_ * sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "[GPU] alloc results failed\n"); return false; }

    err = cudaMalloc(&d_scores_, allocStates_ * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "[GPU] alloc scores failed\n"); return false; }

    err = cudaMalloc(&d_sampled_, allocStates_ * sizeof(uint64_t));
    if (err != cudaSuccess) { fprintf(stderr, "[GPU] alloc sampled failed\n"); return false; }

    err = cudaMalloc(&d_weights_, 36 * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "[GPU] alloc weights failed\n"); return false; }

    err = cudaMalloc(&d_params_, sizeof(EvalParams));
    if (err != cudaSuccess) { fprintf(stderr, "[GPU] alloc params failed\n"); return false; }

    ready_ = true;
    printf("[GPU] Initialized. Buffer capacity: %zu states (%.1f MB)\n",
           allocStates_, allocStates_ * GPU_STATE_SIZE / (1024.0 * 1024.0));
    return true;
}

void GpuEngine::shutdown() {
    if (d_states_) cudaFree(d_states_);
    if (d_results_) cudaFree(d_results_);
    if (d_scores_) cudaFree(d_scores_);
    if (d_sampled_) cudaFree(d_sampled_);
    if (d_weights_) cudaFree(d_weights_);
    if (d_params_) cudaFree(d_params_);
    d_states_ = d_results_ = d_scores_ = d_sampled_ = d_weights_ = d_params_ = nullptr;
    ready_ = false;
}

GpuRolloutResult GpuEngine::runRollouts(
    const std::vector<GpuGameState>& states,
    int botPlayer
) {
    GpuRolloutResult res{0, 0, 0, 0.0f};
    if (!ready_ || states.empty()) return res;

    int N = (int)states.size();
    if ((size_t)N > allocStates_) N = (int)allocStates_;

    // Upload states
    cudaMemcpy(d_states_, states.data(), N * GPU_STATE_SIZE, cudaMemcpyHostToDevice);

    // Launch rollouts: 256 threads/block, оптимально для 4060 Ti (34 SM)
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Ограничиваем для occupancy на 4060 Ti
    blocks = std::min(blocks, 34 * 16); // 34 SM * 16 blocks/SM

    launchRollouts((const GpuGameState*)d_states_, (int*)d_results_, N, botPlayer, rngSeed_);
    rngSeed_ += 7919; // простое число

    // Reduce
    int *d_w, *d_l, *d_dr;
    cudaMalloc(&d_w, sizeof(int)); cudaMemset(d_w, 0, sizeof(int));
    cudaMalloc(&d_l, sizeof(int)); cudaMemset(d_l, 0, sizeof(int));
    cudaMalloc(&d_dr, sizeof(int)); cudaMemset(d_dr, 0, sizeof(int));

    launchReduce((const int*)d_results_, d_w, d_l, d_dr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&res.wins, d_w, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.losses, d_l, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&res.draws, d_dr, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_w); cudaFree(d_l); cudaFree(d_dr);

    int total = res.wins + res.losses + res.draws;
    res.winRate = (total > 0) ? (float)(res.wins + 0.5f * res.draws) / total : 0.5f;
    return res;
}

std::vector<float> GpuEngine::evaluateBatch(
    const std::vector<GpuGameState>& states,
    int botPlayer
) {
    std::vector<float> scores(states.size(), 0.0f);
    if (!ready_ || states.empty()) return scores;

    int N = (int)states.size();
    cudaMemcpy(d_states_, states.data(), N * GPU_STATE_SIZE, cudaMemcpyHostToDevice);

    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    launchEval((const GpuGameState*)d_states_, (float*)d_scores_, N, botPlayer);
    cudaDeviceSynchronize();

    cudaMemcpy(scores.data(), d_scores_, N * sizeof(float), cudaMemcpyDeviceToHost);
    return scores;
}

std::vector<uint64_t> GpuEngine::sampleOpponentHands(
    uint64_t knownCards,
    const float weights[36],
    int handSize,
    int count
) {
    std::vector<uint64_t> hands(count, 0);
    if (!ready_ || count <= 0) return hands;

    int N = std::min(count, (int)allocStates_);
    cudaMemcpy(d_weights_, weights, 36 * sizeof(float), cudaMemcpyHostToDevice);

    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    launchSample((uint64_t*)d_sampled_, knownCards, (const float*)d_weights_, handSize, N, rngSeed_);
    rngSeed_ += 104729;
    cudaDeviceSynchronize();

    cudaMemcpy(hands.data(), d_sampled_, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    return hands;
}

void GpuEngine::setEvalParams(const EvalParams& params) {
    if (!ready_) return;
    uploadEvalParams(params);
}

GpuEngine::DeviceInfo GpuEngine::getDeviceInfo() const {
    DeviceInfo info{};
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    strncpy(info.name, prop.name, 255);
    info.computeMajor = prop.major;
    info.computeMinor = prop.minor;
    info.totalMem = prop.totalGlobalMem;
    info.smCount = prop.multiProcessorCount;
    info.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    return info;
}