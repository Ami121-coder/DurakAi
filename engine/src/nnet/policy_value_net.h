// ============================================================================
// policy_value_net.h.patched — добавлен OnnxNet интерфейс.
//
// Что меняется:
//   1. PolicyValueNet получает метод evaluateBatch() для батчевой оценки
//      (для скорости в self-play: один forward на N позиций вместо N forwards).
//   2. Добавлен класс OnnxNet — реализация через ONNX Runtime (CUDA EP).
//      Реализация в onnx_net.cpp. Принимает путь к .onnx файлу.
//   3. Task 3: evaluate() принимает Knowledge — для доступа к oppProbs.
// ============================================================================

#pragma once

#include "../match.h"
#include "../move.h"
#include "../knowledge.h"

#include <memory>
#include <utility>
#include <vector>

namespace durakk {

struct PVResult {
    std::vector<std::pair<Move, float>> policy;
    float value = 0.5f;
};

class PolicyValueNet {
public:
    virtual ~PolicyValueNet() = default;
    // Task 3: добавлен параметр Knowledge — нужен для доступа к oppProbs.
    virtual PVResult evaluate(const MatchState& s, const Knowledge& k,
                              Player viewpoint) = 0;

    // Батчевая оценка для self-play. Реализация по умолчанию — цикл.
    // OnnxNet переопределит для реального батчинга.
    virtual std::vector<PVResult> evaluateBatch(const std::vector<MatchState>& states,
                                                const std::vector<Knowledge>& knowledges,
                                                Player viewpoint) {
        std::vector<PVResult> out;
        out.reserve(states.size());
        for (size_t i = 0; i < states.size(); ++i) {
            out.push_back(evaluate(states[i], knowledges[i], viewpoint));
        }
        return out;
    }

    virtual bool isReady() const { return true; }
};

// Заглушка — uniform policy, value = 0.5.
class RandomNet : public PolicyValueNet {
public:
    PVResult evaluate(const MatchState& s, const Knowledge& k,
                      Player viewpoint) override;
};

// Реальная сеть через ONNX Runtime + CUDA Execution Provider.
// Реализация в onnx_net.cpp. Принимает путь к .onnx файлу.
class OnnxNet : public PolicyValueNet {
public:
    explicit OnnxNet(const std::string& onnx_path,
                     const std::string& provider = "CUDA",
                     int gpu_device_id = 0);
    ~OnnxNet() override;

    PVResult evaluate(const MatchState& s, const Knowledge& k,
                      Player viewpoint) override;
    std::vector<PVResult> evaluateBatch(const std::vector<MatchState>& states,
                                        const std::vector<Knowledge>& knowledges,
                                        Player viewpoint) override;

    bool isReady() const { return ready_; }

private:
    bool ready_ = false;
    // Реализация спрятана в .cpp, чтобы не тащить onnxruntime.h в хедер.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace durakk
