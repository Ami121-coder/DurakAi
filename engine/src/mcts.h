#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include "bitboard.h"
#include "gpu_engine.h"

// Кодировка хода
constexpr int MOVE_ATTACK   = 0;
constexpr int MOVE_DEFEND   = 1;
constexpr int MOVE_TAKE     = 2;
constexpr int MOVE_BITO     = 3;
constexpr int MOVE_TRANSFER = 4;
constexpr int MOVE_PURSUE   = 5; // подкидывание вдогонку

inline int encodeMove(int cardIdx, int action) { return (cardIdx << 4) | action; }
inline int moveCard(int m) { return m >> 4; }
inline int moveAction(int m) { return m & 0xF; }

struct MctsNode {
    int move = -1;
    int parent = -1;
    int firstChild = -1;
    int nextSibling = -1;
    int visits = 0;
    float totalValue = 0.0f;
    float prior = 0.0f;       // PUCT prior из GPU eval
    uint8_t player = 0;
    bool terminal = false;
    bool expanded = false;
};

struct MctsConfig {
    int maxIterations = 300000;
    int gpuBatchSize = 262144;      // 256K rollout'ов за батч
    int gpuRolloutsPerLeaf = 128;   // rollout'ов на лист
    float puctC = 2.5f;             // PUCT константа
    float evalWeight = 0.4f;        // вес prior vs rollout
    int maxTreeNodes = 3000000;     // ~192 MB (32 GB DDR5)
    int timeLimitMs = 4500;
    bool usePuct = true;            // PUCT (с prior) vs UCB1
};

class MctsEngine {
public:
    MctsEngine() = default;
    ~MctsEngine() = default;

    void init(GpuEngine* gpu);
    void setConfig(const MctsConfig& cfg);

    int decide(
        uint64_t botHand, uint64_t oppHand,
        uint64_t tableAtk, uint64_t tableDef,
        uint64_t deck, int trumpSuit,
        int phase, int attacker, int botPlayer,
        int oppCardCount, int pairsLimit, bool firstTrick,
        const float bayesWeights[36]
    );

    struct SearchStats {
        int totalIterations = 0;
        int gpuRollouts = 0;
        int treeNodes = 0;
        float bestWinRate = 0.5f;
        int timeMs = 0;
    };
    SearchStats getLastStats() const { return lastStats_; }

private:
    GpuEngine* gpu_ = nullptr;
    MctsConfig cfg_;
    std::vector<MctsNode> tree_;
    SearchStats lastStats_{};

    int selectNode(int root);
    int expandNode(int nodeIdx, uint64_t hand, uint64_t oppHand,
                   uint64_t tableAtk, uint64_t tableDef,
                   uint64_t deck, int trump, int phase, int attacker,
                   int player, int oppCardCount, int pairsLimit, bool firstTrick);
    void backpropagate(int nodeIdx, float value);

    std::vector<int> genMoves(uint64_t hand, uint64_t tableAtk, uint64_t tableDef,
                              uint64_t deck, int trump, int phase, int attacker,
                              int player, int oppCardCount, int pairsLimit, bool firstTrick);

    float gpuRolloutBatch(uint64_t hand, uint64_t oppHand,
                          uint64_t tableAtk, uint64_t tableDef,
                          uint64_t deck, int trump, int phase,
                          int attacker, int botPlayer, int pairsLimit,
                          bool firstTrick, int count);

    // PUCT: вычислить prior для детей через GPU eval
    void computePriors(int nodeIdx, uint64_t hand, uint64_t oppHand,
                       uint64_t tableAtk, uint64_t tableDef,
                       uint64_t deck, int trump, int phase, int attacker,
                       int botPlayer, int pairsLimit, bool firstTrick);
};