#pragma once
#include <cstdint>
#include <vector>
#include <memory>

// Forward decl CUDA types
struct GpuGameState;
struct EvalParams;

struct GpuRolloutResult {
    int wins;
    int losses;
    int draws;
    float winRate;
};

class GpuEngine {
public:
    GpuEngine();
    ~GpuEngine();

    bool init(int deviceIndex = 0);
    void shutdown();
    bool isReady() const { return ready_; }

    // Запуск N параллельных rollout'ов
    GpuRolloutResult runRollouts(
        const std::vector<GpuGameState>& states,
        int botPlayer
    );

    // Массовая оценка позиций
    std::vector<float> evaluateBatch(
        const std::vector<GpuGameState>& states,
        int botPlayer
    );

    // Семплирование рук соперника
    std::vector<uint64_t> sampleOpponentHands(
        uint64_t knownCards,
        const float weights[36],
        int handSize,
        int count
    );

    // Установка параметров оценки
    void setEvalParams(const EvalParams& params);

    // Инфо об устройстве
    struct DeviceInfo {
        char name[256];
        int computeMajor, computeMinor;
        size_t totalMem;
        int smCount;
        int maxThreadsPerSM;
    };
    DeviceInfo getDeviceInfo() const;

private:
    bool ready_ = false;
    void* d_states_ = nullptr;
    void* d_results_ = nullptr;
    void* d_scores_ = nullptr;
    void* d_sampled_ = nullptr;
    void* d_weights_ = nullptr;
    void* d_params_ = nullptr;
    size_t allocStates_ = 0;
    size_t allocResults_ = 0;
    unsigned long long rngSeed_ = 12345;
};
