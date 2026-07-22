#include <cuda_runtime.h>
#include "gpu_kernels.cu"

extern "C" {

void launchRollouts(const GpuGameState* states, int* results, int N,
                    int botPlayer, unsigned long long seed, void* stream) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    // RTX 4060 Ti: 34 SM, макс ~16 blocks/SM для occupancy
    blocks = min(blocks, 34 * 16);
    rolloutKernel<<<blocks, tpb, 0, (cudaStream_t)stream>>>(
        states, results, N, botPlayer, seed);
}

void launchEval(const GpuGameState* states, float* scores, int N,
                int botPlayer, void* stream) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    evalKernel<<<blocks, tpb, 0, (cudaStream_t)stream>>>(states, scores, N, botPlayer);
}

void launchSample(uint64_t* sampled, uint64_t known, const float* weights,
                  int handSize, int N, unsigned long long seed, void* stream) {
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    sampleHandsKernel<<<blocks, tpb, 0, (cudaStream_t)stream>>>(
        sampled, known, weights, handSize, N, seed);
}

void launchReduce(const int* results, int* wins, int* losses, int* draws,
                  int N, void* stream) {
    int tpb = 256;
    int blocks = min((N + tpb - 1) / tpb, 128);
    reduceResults<<<blocks, tpb, 0, (cudaStream_t)stream>>>(results, wins, losses, draws, N);
}

void uploadEvalParams(const EvalParams& params) {
    cudaMemcpyToSymbol(d_params, &params, sizeof(EvalParams));
}

} // extern "C"