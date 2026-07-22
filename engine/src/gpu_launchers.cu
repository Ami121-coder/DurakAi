#include <cuda_runtime.h>
#include "gpu_kernels.cu"

// C-обёртки для вызова из C++ (gpu_engine.cpp)
extern "C" {

void launchRollouts(const GpuGameState* states, int* results, int N, int botPlayer, unsigned long long seed) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    rolloutKernel<<<blocks, tpb>>>(states, results, N, botPlayer, seed);
}

void launchEval(const GpuGameState* states, float* scores, int N, int botPlayer) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    evalKernel<<<blocks, tpb>>>(states, scores, N, botPlayer);
}

void launchSample(uint64_t* sampled, uint64_t known, const float* weights, int handSize, int N, unsigned long long seed) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    sampleHandsKernel<<<blocks, tpb>>>(sampled, known, weights, handSize, N, seed);
}

void launchReduce(const int* results, int* wins, int* losses, int* draws, int N) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    blocks = min(blocks, 128);
    reduceResults<<<blocks, tpb>>>(results, wins, losses, draws, N);
}

void uploadEvalParams(const EvalParams& params) {
    cudaMemcpyToSymbol(d_params, &params, sizeof(EvalParams));
}

} // extern "C"