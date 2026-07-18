// ============================================================================
// onnx_net.cpp — обновлён под новый формат модели (FIX #5)
//
// Изменения:
//   1. encodeStateToBuffer теперь использует uint8 → float нормализацию
//      (значения 0..1 вместо 0..255). Модель ожидает нормализованный вход.
//   2. value теперь в [-1, 1] (tanh) — приводим к [0, 1] для ISMCTS.
//   3. Убран мёртвый putMask лямбда-баг.
//   4. Добавлен козырь one-hot в скаляры (4 байта).
// ============================================================================

#include "policy_value_net.h"
#include "../rules_fast.h"
#include "../bitboard.h"
#include "../card.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

namespace durakk {

namespace {

constexpr int kStateSize = 220;
constexpr int kActionSize = 38;

void encodeStateToBuffer(const MatchState& s, Player viewpoint, float* out) {
    Player opp = other(viewpoint);
    float* p = out;

    // 5 масок × 36 = 180 float, значения 0.0 или 1.0
    auto writeMask = [&](CardMask m) {
        for (int i = 0; i < 36; ++i) *p++ = (float)((m >> i) & 1u);
    };

    CardMask trumpMask = 0;
    for (int r = 6; r <= 14; ++r)
        trumpMask |= cardBit(Card{static_cast<Rank>(r), s.trump});

    CardMask atkMask = 0, defMask = 0;
    for (int i = 0; i < s.tableLen; ++i) {
        atkMask |= cardBit(s.table[i].attack);
        if (s.table[i].defended) defMask |= cardBit(s.table[i].defense);
    }

    writeMask(s.hands[toIdx(viewpoint)]);  // [0..35]
    writeMask(trumpMask);                   // [36..71]
    writeMask(atkMask);                     // [72..107]
    writeMask(defMask);                     // [108..143]
    writeMask(s.discard);                   // [144..179]

    // Скаляры [180..219] — нормализованные в [0, 1]
    // Козырь one-hot (4 байта)
    for (int su = 0; su < 4; ++su)
        *p++ = ((int)s.trump == su) ? 1.0f : 0.0f;  // 180-183

    *p++ = (float)s.tableLen / 6.0f;                              // 184
    *p++ = (float)s.undefendedCount() / 6.0f;                    // 185
    *p++ = (float)s.pairsHeadroom() / 6.0f;                      // 186
    *p++ = (float)s.deckRemaining / 36.0f;                       // 187
    *p++ = (float)handSize(s.hands[toIdx(opp)]) / 36.0f;         // 188
    *p++ = (float)handSize(s.hands[toIdx(viewpoint)]) / 36.0f;  // 189
    *p++ = s.firstTrick ? 1.0f : 0.0f;                           // 190
    *p++ = s.transferEnabled ? 1.0f : 0.0f;                      // 191
    *p++ = s.flashEnabled ? 1.0f : 0.0f;                         // 192
    *p++ = (s.phase == MatchPhase::Attack) ? 1.0f : 0.0f;        // 193
    *p++ = (s.phase == MatchPhase::Defense) ? 1.0f : 0.0f;       // 194
    *p++ = (s.turn == viewpoint) ? 1.0f : 0.0f;                  // 195
    *p++ = (s.turn == opp) ? 1.0f : 0.0f;                        // 196
    *p++ = (s.attacker == viewpoint) ? 1.0f : 0.0f;              // 197
    *p++ = (s.attacker == opp) ? 1.0f : 0.0f;                    // 198
    *p++ = 0.0f;  // 199 (reserved)

    // Остаток — нули (200..219)
    while (p < out + kStateSize) *p++ = 0.0f;
}

void computeLegalMask(const MatchState& s, float* out) {
    std::fill(out, out + kActionSize, 0.0f);
    if (s.isGameOver()) return;
    MoveBuffer buf;
    int n = genLegalMoves(s, buf);
    for (int i = 0; i < n; ++i) {
        const Move& m = buf[i];
        int a;
        if (m.action == Action::Take) a = 36;
        else if (m.action == Action::Done || m.action == Action::Pass) a = 37;
        else a = cardIndex(m.card);
        if (a >= 0 && a < kActionSize) out[a] = 1.0f;
    }
}

std::vector<std::pair<Move, float>> buildPolicy(const MatchState& s,
                                                const float* logits,
                                                const float* legal_mask) {
    float masked[kActionSize];
    for (int i = 0; i < kActionSize; ++i) {
        masked[i] = legal_mask[i] > 0.5f ? logits[i] : -1e9f;
    }
    float mx = *std::max_element(masked, masked + kActionSize);
    float sum = 0.0f;
    float probs[kActionSize];
    for (int i = 0; i < kActionSize; ++i) {
        probs[i] = std::exp(masked[i] - mx);
        sum += probs[i];
    }
    if (sum <= 0) sum = 1.0f;
    for (int i = 0; i < kActionSize; ++i) probs[i] /= sum;

    std::vector<std::pair<Move, float>> out;
    out.reserve(kActionSize);
    MoveBuffer buf;
    int n = genLegalMoves(s, buf);
    for (int i = 0; i < n; ++i) {
        const Move& m = buf[i];
        int a;
        if (m.action == Action::Take) a = 36;
        else if (m.action == Action::Done || m.action == Action::Pass) a = 37;
        else a = cardIndex(m.card);
        if (a >= 0 && a < kActionSize && probs[a] > 1e-6f) {
            out.emplace_back(m, probs[a]);
        }
    }
    return out;
}

} // namespace

struct OnnxNet::Impl {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::SessionOptions session_options;

    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "durakk_onnx") {}

    void init(const std::string& path, const std::string& provider, int gpu_id) {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (provider == "CUDA") {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = gpu_id;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 4ULL * 1024 * 1024 * 1024;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } else if (provider == "TensorRT") {
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = gpu_id;
            trt_options.trt_fp16_enable = 1;
            session_options.AppendExecutionProvider_TensorRT(trt_options);
        }

#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), session_options);
#else
        session = Ort::Session(env, path.c_str(), session_options);
#endif
    }
};

OnnxNet::OnnxNet(const std::string& onnx_path,
                 const std::string& provider,
                 int gpu_device_id)
    : impl_(std::make_unique<Impl>()) {
    try {
        impl_->init(onnx_path, provider, gpu_device_id);
        ready_ = true;
        std::fprintf(stderr, "[OnnxNet] Загружена модель %s (provider=%s)\n",
                     onnx_path.c_str(), provider.c_str());
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[OnnxNet] Ошибка загрузки модели: %s\n", e.what());
        ready_ = false;
    }
}

OnnxNet::~OnnxNet() = default;

PVResult OnnxNet::evaluate(const MatchState& s, Player viewpoint) {
    if (!ready_) {
        return RandomNet{}.evaluate(s, viewpoint);
    }

    std::array<float, kStateSize> state_buf;
    std::array<float, kActionSize> mask_buf;
    encodeStateToBuffer(s, viewpoint, state_buf.data());
    computeLegalMask(s, mask_buf.data());

    auto& sess = impl_->session;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> state_shape{1, kStateSize};
    std::array<int64_t, 2> mask_shape{1, kActionSize};

    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        memory_info, state_buf.data(), state_buf.size(),
        state_shape.data(), state_shape.size());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, mask_buf.data(), mask_buf.size(),
        mask_shape.data(), mask_shape.size());

    const char* input_names[] = {"state", "legal_mask"};
    const char* output_names[] = {"policy", "value"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(state_tensor));
    input_tensors.push_back(std::move(mask_tensor));

    auto outputs = sess.Run(Ort::RunOptions{nullptr},
                            input_names, input_tensors.data(), 2,
                            output_names, 2);

    const float* policy_logits = outputs[0].GetTensorData<float>();
    const float* value_out = outputs[1].GetTensorData<float>();

    PVResult res;
    res.policy = buildPolicy(s, policy_logits, mask_buf.data());
    // value в [-1, 1] (tanh) → приводим к [0, 1] для ISMCTS
    float v = value_out[0];
    res.value = (v + 1.0f) * 0.5f;
    if (res.value < 0.0f) res.value = 0.0f;
    if (res.value > 1.0f) res.value = 1.0f;
    return res;
}

std::vector<PVResult> OnnxNet::evaluateBatch(const std::vector<MatchState>& states,
                                             Player viewpoint) {
    if (!ready_ || states.empty()) {
        std::vector<PVResult> out;
        for (const auto& s : states) out.push_back(RandomNet{}.evaluate(s, viewpoint));
        return out;
    }

    const int N = (int)states.size();
    std::vector<float> state_buf(N * kStateSize);
    std::vector<float> mask_buf(N * kActionSize);
    for (int i = 0; i < N; ++i) {
        encodeStateToBuffer(states[i], viewpoint, state_buf.data() + i * kStateSize);
        computeLegalMask(states[i], mask_buf.data() + i * kActionSize);
    }

    auto& sess = impl_->session;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> state_shape{N, kStateSize};
    std::array<int64_t, 2> mask_shape{N, kActionSize};

    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        memory_info, state_buf.data(), state_buf.size(),
        state_shape.data(), state_shape.size());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, mask_buf.data(), mask_buf.size(),
        mask_shape.data(), mask_shape.size());

    const char* input_names[] = {"state", "legal_mask"};
    const char* output_names[] = {"policy", "value"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(state_tensor));
    input_tensors.push_back(std::move(mask_tensor));

    auto outputs = sess.Run(Ort::RunOptions{nullptr},
                            input_names, input_tensors.data(), 2,
                            output_names, 2);

    const float* policy_logits = outputs[0].GetTensorData<float>();
    const float* value_out = outputs[1].GetTensorData<float>();

    std::vector<PVResult> results;
    results.reserve(N);
    for (int i = 0; i < N; ++i) {
        PVResult r;
        r.policy = buildPolicy(states[i],
                               policy_logits + i * kActionSize,
                               mask_buf.data() + i * kActionSize);
        float v = value_out[i];
        r.value = (v + 1.0f) * 0.5f;
        if (r.value < 0.0f) r.value = 0.0f;
        if (r.value > 1.0f) r.value = 1.0f;
        results.push_back(std::move(r));
    }
    return results;
}

} // namespace durakk
