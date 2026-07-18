// ============================================================================
// ismcts.h.patched — PUCT + policy/value net интерфейс внутри ISMCTS.
//
// Что меняется:
//   1. IsmctsLimits получает поля puctC, dirichletAlpha, dirichletEps,
//      rootTemperature, maxRolloutDepth, virtualLoss — для AlphaZero-стиля.
//   2. runIsmcts() принимает опциональный PolicyValueNet* — если задан,
//      узел дерева при расширении получает prior из сети, выбор ребёнка
//      идёт по PUCT (вместо UCB1), а leaf-оценка берётся из value head сети
//      (вместо случайного rollout до конца). Если net=nullptr — старый путь.
//   3. IsmctsResult получает поле rootValue — value head сети в корне.
// ============================================================================

#pragma once

#include "knowledge.h"
#include "match.h"
#include "move.h"
#include "nnet/policy_value_net.h"

#include <atomic>
#include <memory>
#include <vector>
#include <utility>

namespace durakk {

struct IsmctsLimits {
    double timeBudgetSec    = 2.0;
    int    numThreads       = 8;
    // UCB1 (если net=nullptr) — действующая explorationC.
    double explorationC     = 0.7;
    // PUCT (если net!=nullptr).
    double puctC            = 1.4;
    // Dirichlet noise на корне (AlphaZero: alpha=0.03 для шахмат, для Durak
    // с ~10-15 легальных ходов в среднем — alpha≈0.3, eps=0.25).
    double dirichletAlpha   = 0.3;
    double dirichletEps     = 0.25;
    // Температура для корневого выбора хода при self-play.
    // 0.0 = argmax(visits), 1.0 = пропорционально visits, >1 = сглаживание.
    double rootTemperature  = 1.0;
    // Если 0 — rollout до конца партии. Иначе обрываем на этой глубине и
    // оцениваем leaf через value head сети (если есть) или лёгкую эвристику.
    int    maxRolloutDepth  = 0;
    // Virtual losses для tree-parallel MCTS (снижает contention).
    double virtualLoss      = 1.0;
};

struct IsmctsResult {
    Move   move;
    long   playouts = 0;
    double winrate = 0.0;
    double timeMs = 0.0;
    int    rootChildren = 0;
    double rootValue = 0.5;   // value head сети в корне (если был net)
    std::vector<std::pair<Move, double>> rootProbs;
};

// Главный запуск ISMCTS.
//   net — опциональная policy/value сеть. Если nullptr — старый UCB1 + rollout.
IsmctsResult runIsmcts(const MatchState& rootState,
                       const Knowledge& knowledge,
                       const IsmctsLimits& lim,
                       std::atomic<bool>* stopFlag = nullptr,
                       Player viewpoint = Player::Me,
                       PolicyValueNet* net = nullptr);

} // namespace durakk
