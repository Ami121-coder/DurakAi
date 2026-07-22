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
static void updateBayesWeights(const GameState& state) {
    // Карты, которые точно НЕ у соперника:
    // - наши карты
    // - карты на столе
    // - карты, которые соперник не смог побить (вероятно нет)
    uint64_t knownNotOpp = 0;

    // Наши карты
    for (auto& c : state.hands[0]) { // bot = player 0
        knownNotOpp |= bb::cardBit(c.rank, c.suit);
    }

    // Стол
    for (auto& p : state.table) {
        knownNotOpp |= bb::cardBit(p.attack.rank, p.attack.suit);
        if (p.defense.rank >= 0)
            knownNotOpp |= bb::cardBit(p.defense.rank, p.defense.suit);
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
        if (p.defense.rank < 0) { // не побита
            int atkRank = p.attack.rank;
            // Карты того же ранга менее вероятны (уже на столе)
            // Карты ранга atkRank+1 той же масти менее вероятны (мог бы побить)
            int nextRank = atkRank + 1;
            if (nextRank <= 14) {
                int idx = bb::cardIndex(nextRank, p.attack.suit);
                g_bayesWeights[idx] *= 0.3f; // штраф
            }
            // Козырь той же масти менее вероятен
            int trumpIdx = bb::cardIndex(atkRank, state.trumpSuit);
            if (!(knownNotOpp & (1ULL << trumpIdx)))
                g_bayesWeights[trumpIdx] *= 0.5f;
        }
    }
}

// ============================================================
// Конвертация GameState → bitboard
// ============================================================
static uint64_t handToBitboard(const std::vector<Card>& hand) {
    uint64_t mask = 0;
    for (auto& c : hand) mask |= bb::cardBit(c.rank, c.suit);
    return mask;
}

static uint64_t tableAtkToBitboard(const GameState& state) {
    uint64_t mask = 0;
    for (auto& p : state.table) mask |= bb::cardBit(p.attack.rank, p.attack.suit);
    return mask;
}

static uint64_t tableDefToBitboard(const GameState& state) {
    uint64_t mask = 0;
    for (auto& p : state.table)
        if (p.defense.rank >= 0) mask |= bb::cardBit(p.defense.rank, p.defense.suit);
    return mask;
}

static uint64_t deckToBitboard(const GameState& state) {
    // Все карты минус известные
    uint64_t all = (1ULL << 36) - 1;
    uint64_t known = handToBitboard(state.hands[0]);
    known |= tableAtkToBitboard(state);
    known |= tableDefToBitboard(state);
    // Карты соперника неизвестны, но их количество знаем
    // Колода = все - наши - стол - соперник (но соперника не знаем)
    // Аппроксимация: колода = все - наши - стол (с учётом deckCount)
    uint64_t deck = all & ~known;
    // Ограничиваем по deckCount
    int dc = state.deckCount;
    uint64_t result = 0;
    uint64_t tmp = deck;
    for (int i = 0; i < dc && tmp; ++i) {
        int bit = bb::popLowest(tmp);
        result |= (1ULL << bit);
    }
    return result;
}

// ============================================================
// Главная функция: Bot::decide()
// ============================================================
Move Bot::decide(const GameState& state) {
    ensureInit();

    int botPlayer = 0; // бот всегда player 0
    int oppPlayer = 1;

    // Обновляем байесовские веса
    updateBayesWeights(state);

    // Конвертируем в bitboard
    uint64_t botHand = handToBitboard(state.hands[botPlayer]);
    uint64_t tableAtk = tableAtkToBitboard(state);
    uint64_t tableDef = tableDefToBitboard(state);
    uint64_t deck = deckToBitboard(state);
    int trump = state.trumpSuit;
    int phase = (state.phase == Phase::ATTACK) ? 0 : 1;
    int attacker = state.attacker;
    int oppCardCount = (int)state.hands[oppPlayer].size();
    if (oppCardCount == 0) oppCardCount = state.oppCardCount; // из UI

    // Запускаем MCTS
    int encodedMove = g_mcts.decide(
        botHand, 0 /*oppHand unknown*/,
        tableAtk, tableDef, deck,
        trump, phase, attacker, botPlayer,
        oppCardCount, g_bayesWeights
    );

    // Логируем статистику
    auto stats = g_mcts.getLastStats();
    printf("[Bot] Search: %d iter, %d GPU rollouts, %d nodes, %.1f%% WR, %d ms\n",
           stats.totalIterations, stats.gpuRollouts, stats.treeNodes,
           stats.bestWinRate * 100.0f, stats.timeMs);

    // Декодируем ход в Move
    Move result;
    if (encodedMove < 0) {
        // Fallback: первая легальная карта
        result.type = MoveType::ATTACK;
        if (!state.hands[botPlayer].empty())
            result.card = state.hands[botPlayer][0];
        return result;
    }

    int cardIdx = moveCard(encodedMove);
    int action = moveAction(encodedMove);

    result.card.rank = bb::rankOf(cardIdx);
    result.card.suit = bb::suitOf(cardIdx);

    switch (action) {
    case MOVE_ATTACK:   result.type = MoveType::ATTACK; break;
    case MOVE_DEFEND:   result.type = MoveType::DEFEND; break;
    case MOVE_TAKE:     result.type = MoveType::TAKE; break;
    case MOVE_BITO:     result.type = MoveType::BITO; break;
    case MOVE_TRANSFER: result.type = MoveType::TRANSFER; break;
    default:            result.type = MoveType::ATTACK; break;
    }

    return result;
}
