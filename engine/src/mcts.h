#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include "bitboard.h"
#include "gpu_engine.h"

// Узел дерева MCTS
struct MctsNode {
    int move = -1;           // ход, ведущий в этот узел (индекс карты или код действия)
    int parent = -1;
    int firstChild = -1;
    int nextSibling = -1;
    int visits = 0;
    float totalValue = 0.0f;
    int untriedMoves = 0;
    uint8_t player = 0;      // чей ход в этом узле
    bool terminal = false;
    float terminalValue = 0.0f;
};

// Кодировка хода: (cardIdx << 4) | actionType
// actionType: 0=attack, 1=defend, 2=take, 3=pass(bito), 4=transfer
constexpr int MOVE_ATTACK = 0;
constexpr int MOVE_DEFEND = 1;
constexpr int MOVE_TAKE   = 2;
constexpr int MOVE_BITO   = 3;
constexpr int MOVE_TRANSFER = 4;

inline int encodeMove(int cardIdx, int action) { return (cardIdx << 4) | action; }
inline int moveCard(int m) { return m >> 4; }
inline int moveAction(int m) { return m & 0xF; }

struct MctsConfig {
    int maxIterations = 500000;     // итераций на CPU-дереве
    int gpuBatchSize = 262144;      // rollout'ов за один GPU-вызов (256K, ~16MB)
    int gpuRolloutsPerLeaf = 64;    // rollout'ов на один лист
    float ucbC = 1.41f;             // UCB1 константа
    float evalWeight = 0.3f;        // вес эвристики vs rollout
    int maxTreeNodes = 2000000;     // макс узлов (32 GB RAM позволяет)
    int timeLimitMs = 4500;         // лимит времени на ход
    int cpuThreads = 10;            // потоков CPU (7500F: 12 threads, 2 оставляем)
};

class MctsEngine {
public:
    MctsEngine();
    ~MctsEngine();

    void init(GpuEngine* gpu);
    void setConfig(const MctsConfig& cfg);

    // Главный вход: вернуть лучший ход
    // state: текущее состояние (руки, стол, колода, козырь)
    // botPlayer: 0 или 1
    // Возвращает: закодированный ход (encodeMove)
    int decide(
        uint64_t botHand,
        uint64_t oppHand,       // может быть 0 если неизвестна
        uint64_t tableAtk,
        uint64_t tableDef,
        uint64_t deck,
        int trumpSuit,
        int phase,              // 0=attack, 1=defend
        int attacker,
        int botPlayer,
        int oppCardCount,
        const float bayesWeights[36]  // веса для семплинга
    );

    // Статистика последнего поиска
    struct SearchStats {
        int totalIterations;
        int gpuRollouts;
        int treeNodes;
        float bestWinRate;
        int timeMs;
    };
    SearchStats getLastStats() const { return lastStats_; }

private:
    GpuEngine* gpu_ = nullptr;
    MctsConfig cfg_;
    std::vector<MctsNode> tree_;
    SearchStats lastStats_{};

    // Внутренние методы
    int selectNode(int nodeIdx);
    int expandNode(int nodeIdx, uint64_t hand, uint64_t oppHand,
                   uint64_t tableAtk, uint64_t tableDef,
                   uint64_t deck, int trump, int phase, int attacker);
    float simulate(int nodeIdx, uint64_t hand, uint64_t oppHand,
                   uint64_t tableAtk, uint64_t tableDef,
                   uint64_t deck, int trump, int phase, int attacker, int botPlayer);
    void backpropagate(int nodeIdx, float value);

    // Генерация легальных ходов
    std::vector<int> genMoves(uint64_t hand, uint64_t tableAtk, uint64_t tableDef,
                              uint64_t deck, int trump, int phase, int attacker,
                              int player, int oppCardCount);

    // Применение хода к состоянию
    void applyMove(int move, uint64_t& hand, uint64_t& oppHand,
                   uint64_t& tableAtk, uint64_t& tableDef,
                   uint64_t& deck, int& phase, int& attacker, int& deckCnt);

    // GPU-батч rollout'ов
    float gpuRolloutBatch(uint64_t hand, uint64_t oppHand,
                          uint64_t tableAtk, uint64_t tableDef,
                          uint64_t deck, int trump, int phase,
                          int attacker, int botPlayer, int count);
};
