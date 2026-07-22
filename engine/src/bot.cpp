#include "bot.h"
#include "bitboard.h"
#include "mcts.h"
#include "gpu_engine.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstdio>

// ============================================================
// Глобальные объекты (живут на время процесса)
// ============================================================
static GpuEngine g_gpu;
static MctsEngine g_mcts;
static bool g_initialized = false;
static float g_bayesWeights[36]; // байесовские веса для семплинга

// Параметры оценки (тюнятся CMA-ES или вручную)
static EvalParams g_evalParams = {
    .w_hand_count   = 3.0f,
    .w_trump_count  = 5.0f,
    .w_high_trump   = 8.0f,
    .w_rank_spread  = 2.0f,
    .w_low_cards    = 1.5f,
    .w_opp_trumps   = 4.0f,
    .w_deck_factor  = 0.5f,
    .w_table_control = 3.0f
};

// ============================================================
// Инициализация (вызывается один раз при старте)
// ============================================================
static void ensureInit() {
    if (g_initialized) return;

    // Инициализация GPU
    if (g_gpu.init(0)) {
        g_gpu.setEvalParams(g_evalParams);
        printf("[Bot] GPU initialized successfully\n");
    } else {
        printf("[Bot] GPU init failed, falling back to CPU-only\n");
    }

    // Инициализация MCTS
    MctsConfig cfg;
    cfg.maxIterations = 500000;
    cfg.gpuBatchSize = 262144;      // 256K rollout'ов за батч
    cfg.gpuRolloutsPerLeaf = 64;
    cfg.ucbC = 1.41f;
    cfg.evalWeight = 0.3f;
    cfg.maxTreeNodes = 2000000;     // ~128 MB RAM (32 GB DDR5 позволяет)
    cfg.timeLimitMs = 4500;         // 4.5 сек на ход
    cfg.cpuThreads = 10;            // 7500F: 12 потоков, 2 на систему

    g_mcts.init(&g_gpu);
    g_mcts.setConfig(cfg);

    // Инициализация байесовских весов (равномерные)
    for (int i = 0; i < 36; ++i) g_bayesWeights[i] = 1.0f;

    g_initialized = true;
    printf("[Bot] MCTS engine ready (GPU: %s)\n", g_gpu.isReady() ? "ON" : "OFF");
}

// ============================================================
// Обновление байесовских весов по истории ходов
// ============================================================
static void updateBayesWeights(const durakk::GameState& state) {
    // Карты, которые точно НЕ у соперника:
    // - наши карты
    // - карты на столе
    // - карты, которые соперник не смог побить (вероятно нет)
    uint64_t knownNotOpp = 0;

    // Наши карты
    for (auto& c : state.hands[0]) { // bot = player 0
        knownNotOpp |= bb::cardBit(c.rank, static_cast<int>(c.suit));
    }

    // Стол
    for (auto& p : state.table) {
        knownNotOpp |= bb::cardBit(p.attack.rank, static_cast<int>(p.attack.suit));
        if (p.defended)
            knownNotOpp |= bb::cardBit(p.defense.rank, static_cast<int>(p.defense.suit));
    }

    // Обновляем веса: известные карты → вес 0
    for (int i = 0; i < 36; ++i) {
        if (knownNotOpp & (1ULL << i)) {
            g_bayesWeights[i] = 0.0f;
        } else {
            // Базовый вес + бонус за "вероятность"
            g_bayesWeights[i] = 1.0f;
        }
    }

    // Эвристика: если соперник не побил карту ранга R — вероятно нет карт ранга R+
    for (auto& p : state.table) {
        if (!p.defended) { // не побита
            int atkRank = p.attack.rank;
            for (int r = atkRank; r <= 14; ++r) {
                for (int s = 0; s < 4; ++s) {
                    int idx = bb::cardIndex(r, s);
                    g_bayesWeights[idx] *= 0.5f;
                }
            }
        }
    }
}

namespace durakk {

Bot::Bot() {
    ensureInit();
}

Move Bot::decide(const GameState& s, const Knowledge& k,
                 const SearchSettings& settings, DecisionStats* statsOut) {
    ensureInit();
    updateBayesWeights(s);

    uint64_t botHand = k.myHand;
    uint64_t tableAtk = 0;
    uint64_t tableDef = 0;

    for (const auto& pair : s.table) {
        tableAtk |= bb::cardBit(pair.attack.rank, static_cast<int>(pair.attack.suit));
        if (pair.defended) {
            tableDef |= bb::cardBit(pair.defense.rank, static_cast<int>(pair.defense.suit));
        }
    }

    uint64_t deck = 0;
    int trumpSuit = static_cast<int>(s.deck.trump);
    int phase = (s.phase == Phase::Attack) ? 0 : 1;
    int attacker = (s.attacker == Side::Me) ? 0 : 1;
    int botPlayer = 0; // Мы играем за 0
    int oppCardCount = k.oppHandCount;

    int moveEncoded = g_mcts.decide(
        botHand, 0, tableAtk, tableDef, deck,
        trumpSuit, phase, attacker, botPlayer,
        oppCardCount, g_bayesWeights
    );

    Move m{};
    if (moveEncoded < 0) {
        m.action = Action::Pass;
        m.reason = "No legal moves";
        return m;
    }

    int cardIdx = moveCard(moveEncoded);
    int actionType = moveAction(moveEncoded);

    m.card = Card(bb::rankOf(cardIdx), static_cast<Suit>(bb::suitOf(cardIdx)));

    switch (actionType) {
    case MOVE_ATTACK:
        m.action = Action::Attack;
        m.reason = "MCTS Attack";
        break;
    case MOVE_DEFEND:
        m.action = Action::Defend;
        m.reason = "MCTS Defend";
        for (const auto& pair : s.table) {
            if (!pair.defended) {
                m.target = pair.attack;
                m.hasTarget = true;
                break;
            }
        }
        break;
    case MOVE_TAKE:
        m.action = Action::Take;
        m.reason = "MCTS Take";
        break;
    case MOVE_BITO:
        m.action = Action::Done;
        m.reason = "MCTS Bito";
        break;
    case MOVE_TRANSFER:
        m.action = Action::Transfer;
        m.reason = "MCTS Transfer";
        break;
    default:
        m.action = Action::Pass;
        m.reason = "MCTS Pass";
        break;
    }

    if (statsOut) {
        SearchStats stats = g_mcts.getLastStats();
        statsOut->mode = g_gpu.isReady() ? "MCTS+GPU" : "MCTS";
        statsOut->playouts = stats.gpuRollouts;
        statsOut->winrate = stats.bestWinRate;
        statsOut->timeMs = stats.timeMs;
    }

    return m;
}

} // namespace durakk