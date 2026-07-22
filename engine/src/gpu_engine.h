#pragma once
#include <cstdint>
#include <vector>
#include <functional>

struct GpuGameState;
struct EvalParams;

struct GpuRolloutResult {
    int wins = 0, losses = 0, draws = 0;
    float winRate = 0.5f;
};

class GpuEngine {
public:
    GpuEngine();
    ~GpuEngine();

    bool init(int deviceIndex = 0);
    void shutdown();
    bool isReady() const { return ready_; }

    // Синхронный batch rollout
    GpuRolloutResult runRollouts(const std::vector<GpuGameState>& states, int botPlayer);

    // Асинхронный rollout (double buffering)
    void runRolloutsAsync(const std::vector<GpuGameState>& states, int botPlayer,
                          std::function<void(GpuRolloutResult)> callback);
    bool pollAsync(GpuRolloutResult& out);

    // Batch evaluation (для PUCT prior)
    std::vector<float> evaluateBatch(const std::vector<GpuGameState>& states, int botPlayer);

    // Sampling рук соперника
    std::vector<uint64_t> sampleOpponentHands(uint64_t knownCards,
                                              const float weights[36],
                                              int handSize, int count);

    void setEvalParams(const EvalParams& params);

    struct DeviceInfo {
        char name[256];
        int smCount;
        size_t totalMem;
    };
    DeviceInfo getDeviceInfo() const;

private:
    bool ready_ = false;

    // Преаллоцированные буферы (ФИКС: никаких malloc/free в hot path)
    void* d_states_[2] = {};    // double buffer
    void* d_results_[2] = {};
    void* d_scores_ = nullptr;
    void* d_sampled_ = nullptr;
    void* d_weights_ = nullptr;
    void* d_wins_ = nullptr;    // преаллоцированные счётчики
    void* d_losses_ = nullptr;
    void* d_draws_ = nullptr;

    // CUDA streams для overlap
    void* streams_[2] = {};     // cudaStream_t

    size_t capacity_ = 0;       // размер буферов (в состояниях)
    int curBuf_ = 0;            // текущий буфер для async
    bool asyncPending_ = false;

    unsigned long long rngSeed_ = 42;

    void ensureCapacity(size_t n);
};