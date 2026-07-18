// ============================================================================
// onnx_net.cpp — реализация OnnxNet через ONNX Runtime + CUDA EP.
//
// Зависимости (CMakeLists.txt.patched):
//   find_package(onnxruntime CONFIG REQUIRED)
//   target_link_libraries(durakk_engine PRIVATE onnxruntime)
//
// Формат .onnx файла (экспортируется из PyTorch см. export_onnx.py):
//   Входы:
//     "state"      — float32[batch, 220]  (или uint8[batch, 220] — лучше float32)
//     "legal_mask" — float32[batch, 38]   (для маскирования policy)
//   Выходы:
//     "policy"     — float32[batch, 38]   (логиты)
//     "value"      — float32[batch, 1]    (в [0, 1] через sigmoid или в [-1,1] через tanh)
//
// Для inference в C++ мы:
//   1. Кодируем MatchState в 220 float.
//   2. Вычисляем legal_mask (38 float).
//   3. Прогоняем через onnxruntime::Ort::Session.
//   4. Masked softmax над policy logits.
//   5. value отображаем в [0,1].
// ============================================================================

#include "policy_value_net.h"
#include "rules_fast.h"
#include "bitboard.h"
#include "card.h"

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

// ---- Кодирование состояния в 220 float (зеркало bindings.cpp::encodeState) ----
void encodeStateToBuffer(const MatchState& s, Player viewpoint, float* out) {
    Player opp = other(viewpoint);
    CardMask trumpMask = 0;
    for (int r = 6; r <= 14; ++r)
        trumpMask |= cardBit(Card{static_cast<Rank>(r), s.trump});

    CardMask atkMask = 0, defMask = 0;
    for (int i = 0; i < s.tableLen; ++i) {
        atkMask |= cardBit(s.table[i].attack);
        if (s.table[i].defended) defMask |= cardBit(s.table[i].defense);
    }

    auto putMask = [&](CardMask m) {
        for (int i = 0; i < 36; ++i) out[i] = (float)((m >> i) & 1u);
        out += 36;
    };
    // Внимание: putMask мутирует локальную копию out — исправим.
    float* p = out;
    for (int i = 0; i < 36; ++i) p[i] = (float)((s.hands[toIdx(viewpoint)] >> i) & 1u);
    p += 36;
    for (int i = 0; i < 36; ++i) p[i] = (float)((trumpMask >> i) & 1u);
    p += 36;
    for (int i = 0; i < 36; ++i) p[i] = (float)((atkMask >> i) & 1u);
    p += 36;
    for (int i = 0; i < 36; ++i) p[i] = (float)((defMask >> i) & 1u);
    p += 36;
    for (int i = 0; i < 36; ++i) p[i] = (float)((s.discard >> i) & 1u);
    p += 36;
    // p сейчас указывает на out[180] — скалярные фичи (40 байт).
    p[0]  = (float)s.tableLen / 6.0f;
    p[1]  = (float)s.undefendedCount() / 6.0f;
    p[2]  = (float)s.pairsHeadroom() / 6.0f;
    p[3]  = (float)s.deckRemaining / 36.0f;
    p[4]  = (float)handSize(s.hands[toIdx(opp)]) / 36.0f;
    p[5]  = (float)handSize(s.hands[toIdx(viewpoint)]) / 36.0f;
    p[6]  = s.firstTrick ? 1.0f : 0.0f;
    p[7]  = s.transferEnabled ? 1.0f : 0.0f;
    p[8]  = s.flashEnabled ? 1.0f : 0.0f;
    p[9]  = (s.phase == MatchPhase::Attack) ? 1.0f : 0.0f;
    p[10] = (s.phase == MatchPhase::Defense) ? 1.0f : 0.0f;
    p[11] = (s.turn == viewpoint) ? 1.0f : 0.0f;
    p[12] = (s.turn == opp) ? 1.0f : 0.0f;
    p[13] = (s.attacker == viewpoint) ? 1.0f : 0.0f;
    p[14] = (s.attacker == opp) ? 1.0f : 0.0f;
    for (int i = 15; i < 40; ++i) p[i] = 0.0f;
}

// ---- Вычисление legal_mask (38 float) ----
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

// ---- Преобразование логитов + legal_mask → policy по ходам ----
std::vector<std::pair<Move, float>> buildPolicy(const MatchState& s,
                                                const float* logits,
                                                const float* legal_mask) {
    // Masked softmax: logit[i] -= 1e9 если legal_mask[i] == 0.
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

    // Сопоставим индексы ходам (зеркало moveToActionIndex).
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

// ---- Impl: всё, что зависит от onnxruntime ----
struct OnnxNet::Impl {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "durakk_onnx") {}

    void init(const std::string& path, const std::string& provider, int gpu_id) {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (provider == "CUDA") {
#ifdef _WIN32
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = gpu_id;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 4ULL * 1024 * 1024 * 1024; // 4 GB лимит
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = gpu_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
        } else if (provider == "TensorRT") {
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = gpu_id;
            trt_options.trt_fp16_enable = 1;  // FP16 inference
            session_options.AppendExecutionProvider_TensorRT(trt_options);
        } else {
            // CPU fallback.
        }

        session = Ort::Session(env, path.c_str(), session_options);
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
        // Fallback на uniform policy.
        return RandomNet{}.evaluate(s, viewpoint);
    }

    // Подготовим входные тензоры [1, 220] и [1, 38].
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

    // Имена входов/выходов — фиксируем (см. export_onnx.py).
    const char* input_names[] = {"state", "legal_mask"};
    const char* output_names[] = {"policy", "value"};

    auto outputs = sess.Run(Ort::RunOptions{nullptr},
                            input_names, {&state_tensor, &mask_tensor}, 2,
                            output_names, 2);

    const float* policy_logits = outputs[0].GetTensorData<float>();
    const float* value_out = outputs[1].GetTensorData<float>();

    PVResult res;
    res.policy = buildPolicy(s, policy_logits, mask_buf.data());
    // value в [0,1] (если модель училась на sigmoid) — берём как есть.
    res.value = value_out[0];
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

    // Батч: [N, 220] + [N, 38] → [N, 38] + [N, 1].
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

    auto outputs = sess.Run(Ort::RunOptions{nullptr},
                            input_names, {&state_tensor, &mask_tensor}, 2,
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
        r.value = value_out[i];
        if (r.value < 0.0f) r.value = 0.0f;
        if (r.value > 1.0f) r.value = 1.0f;
        results.push_back(std::move(r));
    }
    return results;
}

} // namespace durakk
