#pragma once

#include "game_state.h"
#include "knowledge.h"
#include "match.h"
#include "move.h"
#include "nnet/policy_value_net.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace durakk {

// Уровень силы
enum class Strength { Fast, Normal, Deep };

// Параметры силы, общие для ISMCTS и эндшпиля.
struct SearchSettings {
    Strength strength = Strength::Normal;
    int numThreads = 8;
};

// Статистика последнего поиска (для отображения в UI).
struct DecisionStats {
    std::string mode;       // "ISMCTS" | "Endgame" | "EarlyEndgame" | "Fallback" | "Ponder"
    long   playouts = 0;
    double winrate = 0.0;   // ISMCTS
    int    depthReached = 0;// эндшпиль
    bool   solved = false;  // эндшпиль: найден форсированный результат
    double timeMs = 0.0;
};

// Бот-советник: маршрутизирует между ISMCTS (фаза с колодой) и минимаксом
// (эндшпиль, когда колода пуста), используя Knowledge (абсолютную память).
class Bot {
public:
    Bot();
    ~Bot();

    // Полный путь: строит MatchState + Knowledge из GameState, запускает
    // соответствующий решатель, возвращает ход и статистику.
    // `strength` — таймаут/число потоков.
    Move decide(const GameState& s, const Knowledge& k,
                const SearchSettings& settings,
                DecisionStats* statsOut = nullptr);

    // Доступ к сети (чтобы main.cpp мог подменить RandomNet на OnnxNet позже).
    PolicyValueNet* net() { return net_.get(); }

    // True только если загрузилась РЕАЛЬНАЯ нейросеть (OnnxNet с готовой моделью).
    // RandomNet/заглушка сюда НЕ считаются — в decide() в этом случае передаём
    // nullptr в ISMCTS, чтобы поиск шёл по чистой математике (UCB1 + rollout),
    // а не деградировал на uniform priors + value=0.5.
    bool hasRealNet() const { return hasRealNet_; }

    // Task 6: Pondering API.
    // После выдачи хода ботом — вызвать startPondering() с тем же состоянием,
    // чтобы в фоне предрассчитать наиболее вероятные ответы соперника.
    // Когда соперник сходил — вызвать checkPonderCache(state) с НОВЫМ состоянием.
    // Если предсказание совпало — вернётся готовый ход (за 0ms).
    void startPondering(const GameState& s, const Knowledge& k,
                        const SearchSettings& settings);
    Move checkPonderCache(const GameState& s, const Knowledge& k,
                          DecisionStats* statsOut = nullptr);
    void stopPondering();

private:
    std::unique_ptr<PolicyValueNet> net_;
    bool hasRealNet_ = false;

    // Task 6: Pondering state.
    struct PonderEntry {
        Move move;
        DecisionStats stats;
        // Хеши всех предсказанных состояний, для которых считали ответ.
        // Ключ — Zobrist-like hash (упрощённый: myHand + table + trump + turn).
        uint64_t predictedStateHash = 0;
    };
    std::mutex ponderMtx_;
    std::thread ponderThread_;
    std::atomic<bool> ponderStopFlag_{false};
    std::atomic<bool> ponderActive_{false};
    // Кеш предрассчитанных ответов: ключ = hash состояния (после хода соперника).
    std::unordered_map<uint64_t, PonderEntry> ponderCache_;

    // Вспомогательный метод: построить упрощённый хеш состояния для pondering.
    // Используется только для cache lookup, не для TT.
    static uint64_t ponderHash(const GameState& s, const Knowledge& k);

    void ponderWorker(const GameState s, const Knowledge k,
                      const SearchSettings settings);
};

} // namespace durakk
