#include "mcts.h"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <random>
#include <cstdio>

void MctsEngine::init(GpuEngine* gpu) {
    gpu_ = gpu;
    tree_.reserve(cfg_.maxTreeNodes);
}

void MctsEngine::setConfig(const MctsConfig& cfg) {
    cfg_ = cfg;
    tree_.reserve(cfg_.maxTreeNodes);
}

int MctsEngine::decide(
    uint64_t botHand, uint64_t oppHand,
    uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trumpSuit,
    int phase, int attacker, int botPlayer,
    int oppCardCount, int pairsLimit, bool firstTrick,
    const float bayesWeights[36]
) {
    auto t0 = std::chrono::steady_clock::now();
    tree_.clear();

    // Корень
    MctsNode root{};
    root.player = (uint8_t)botPlayer;
    tree_.push_back(root);

    int totalIter = 0, totalGpu = 0;
    bool hiddenOpp = (oppHand == 0 && oppCardCount > 0);
    std::mt19937 rng(12345);

    while (totalIter < cfg_.maxIterations) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        if (elapsed >= cfg_.timeLimitMs) break;
        if ((int)tree_.size() >= cfg_.maxTreeNodes - 100) break;

        // Семплирование руки соперника
        uint64_t curOpp = oppHand;
        if (hiddenOpp) {
            uint64_t known = botHand | tableAtk | tableDef;
            uint64_t pool = ((1ULL << 36) - 1) & ~known;
            curOpp = 0;
            // Weighted sampling на CPU
            std::vector<std::pair<float,int>> weighted;
            float totalW = 0;
            for (int i = 0; i < 36; ++i) {
                if (pool & (1ULL << i)) {
                    weighted.push_back({bayesWeights[i], i});
                    totalW += bayesWeights[i];
                }
            }
            std::uniform_real_distribution<float> dist(0, totalW);
            for (int c = 0; c < oppCardCount && !weighted.empty(); ++c) {
                float r = dist(rng);
                float acc = 0;
                for (int j = 0; j < (int)weighted.size(); ++j) {
                    acc += weighted[j].first;
                    if (acc >= r) {
                        curOpp |= (1ULL << weighted[j].second);
                        totalW -= weighted[j].first;
                        weighted.erase(weighted.begin() + j);
                        break;
                    }
                }
            }
        }

        // 1. Selection (PUCT)
        int node = selectNode(0);

        // 2. Expansion + GPU prior
        if (!tree_[node].expanded && !tree_[node].terminal) {
            uint64_t h = (tree_[node].player == botPlayer) ? botHand : curOpp;
            uint64_t oh = (tree_[node].player == botPlayer) ? curOpp : botHand;

            int child = expandNode(node, h, oh, tableAtk, tableDef, deck,
                                   trumpSuit, phase, attacker,
                                   tree_[node].player, oppCardCount,
                                   pairsLimit, firstTrick);

            // PUCT: вычисляем prior для детей через GPU eval
            if (cfg_.usePuct && gpu_ && gpu_->isReady() && child >= 0) {
                computePriors(node, h, oh, tableAtk, tableDef, deck,
                              trumpSuit, phase, attacker, botPlayer,
                              pairsLimit, firstTrick);
            }
            node = (child >= 0) ? child : node;
        }

        // 3. Simulation (GPU rollout)
        float value;
        if (tree_[node].terminal) {
            value = 0.5f; // терминальный узел
        } else {
            uint64_t h = (tree_[node].player == botPlayer) ? botHand : curOpp;
            uint64_t oh = (tree_[node].player == botPlayer) ? curOpp : botHand;
            value = gpuRolloutBatch(h, oh, tableAtk, tableDef, deck,
                                    trumpSuit, phase, attacker, botPlayer,
                                    pairsLimit, firstTrick,
                                    cfg_.gpuRolloutsPerLeaf);
            totalGpu += cfg_.gpuRolloutsPerLeaf;
        }

        // 4. Backpropagation
        backpropagate(node, value);
        totalIter++;
    }

    // Лучший ход: max visits
    int bestChild = -1, bestVisits = -1;
    for (int c = tree_[0].firstChild; c >= 0; c = tree_[c].nextSibling) {
        if (tree_[c].visits > bestVisits) {
            bestVisits = tree_[c].visits;
            bestChild = c;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    lastStats_ = {totalIter, totalGpu, (int)tree_.size(),
                  (bestChild >= 0) ? tree_[bestChild].totalValue / std::max(1, tree_[bestChild].visits) : 0.5f,
                  (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()};

    return (bestChild >= 0) ? tree_[bestChild].move : -1;
}

int MctsEngine::selectNode(int root) {
    int cur = root;
    while (cur >= 0 && cur < (int)tree_.size()) {
        if (tree_[cur].terminal) return cur;
        if (!tree_[cur].expanded) return cur;
        if (tree_[cur].firstChild < 0) return cur;

        float bestScore = -1e9f;
        int bestChild = -1;
        float parentVisits = (float)std::max(1, tree_[cur].visits);
        float sqrtParent = sqrtf(parentVisits);

        for (int c = tree_[cur].firstChild; c >= 0; c = tree_[c].nextSibling) {
            float childVisits = (float)std::max(1, tree_[c].visits);
            float exploit = tree_[c].totalValue / childVisits;
            float score;

            if (cfg_.usePuct && tree_[c].prior > 0.0f) {
                // PUCT: exploit + C * prior * sqrt(parent) / (1 + child)
                score = exploit + cfg_.puctC * tree_[c].prior * sqrtParent / (1.0f + childVisits);
            } else {
                // UCB1 fallback
                score = exploit + cfg_.puctC * sqrtf(logf(parentVisits) / childVisits);
            }

            if (score > bestScore) { bestScore = score; bestChild = c; }
        }
        cur = bestChild;
    }
    return cur;
}

int MctsEngine::expandNode(
    int nodeIdx, uint64_t hand, uint64_t oppHand,
    uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trump, int phase, int attacker,
    int player, int oppCardCount, int pairsLimit, bool firstTrick
) {
    if (nodeIdx < 0 || nodeIdx >= (int)tree_.size()) return -1;
    if (tree_[nodeIdx].terminal || tree_[nodeIdx].expanded) return nodeIdx;

    auto moves = genMoves(hand, tableAtk, tableDef, deck, trump, phase, attacker,
                          player, oppCardCount, pairsLimit, firstTrick);

    if (moves.empty()) {
        tree_[nodeIdx].terminal = true;
        tree_[nodeIdx].expanded = true;
        return nodeIdx;
    }

    int prevChild = -1;
    for (int m : moves) {
        if ((int)tree_.size() >= cfg_.maxTreeNodes) break;

        MctsNode child{};
        child.move = m;
        child.parent = nodeIdx;
        child.player = (uint8_t)(1 - player);

        int childIdx = (int)tree_.size();
        tree_.push_back(child);

        if (prevChild < 0) tree_[nodeIdx].firstChild = childIdx;
        else tree_[prevChild].nextSibling = childIdx;
        prevChild = childIdx;
    }

    tree_[nodeIdx].expanded = true;
    return tree_[nodeIdx].firstChild;
}

void MctsEngine::backpropagate(int nodeIdx, float value) {
    int cur = nodeIdx;
    while (cur >= 0 && cur < (int)tree_.size()) {
        tree_[cur].visits++;
        tree_[cur].totalValue += (tree_[cur].player == 0) ? value : (1.0f - value);
        cur = tree_[cur].parent;
    }
}

std::vector<int> MctsEngine::genMoves(
    uint64_t hand, uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trump, int phase, int attacker,
    int player, int oppCardCount, int pairsLimit, bool firstTrick
) {
    std::vector<int> moves;
    if (hand == 0) return moves;

    if (phase == 0 && player == attacker) {
        uint64_t tmp = hand;
        while (tmp) {
            int idx = bb::popLowest(tmp);
            moves.push_back(encodeMove(idx, MOVE_ATTACK));
        }
        if (tableAtk != 0 && (tableAtk & ~tableDef) == 0) {
            moves.push_back(encodeMove(0, MOVE_BITO));
        }
    } else if (phase == 1 && player != attacker) {
        uint64_t undefended = tableAtk & ~tableDef;
        if (undefended == 0) {
            moves.push_back(encodeMove(0, MOVE_BITO));
            return moves;
        }

        uint64_t atkTmp = undefended;
        while (atkTmp) {
            int atkIdx = bb::popLowest(atkTmp);
            uint64_t hTmp = hand;
            bool canBeat = false;
            while (hTmp) {
                int defIdx = bb::popLowest(hTmp);
                if (bb::beats(atkIdx, defIdx, trump)) {
                    moves.push_back(encodeMove(defIdx, MOVE_DEFEND));
                    canBeat = true;
                }
            }
            if (!canBeat) {
                moves.push_back(encodeMove(0, MOVE_TAKE));
                return moves;
            }
            break;
        }

        if (tableDef == 0 && tableAtk != 0) {
            int atkRank = bb::rankOf(bb::lowestBit(tableAtk));
            bool allSameRank = true;
            uint64_t tTmp = tableAtk;
            while (tTmp) {
                int ti = bb::popLowest(tTmp);
                if (bb::rankOf(ti) != atkRank) { allSameRank = false; break; }
            }
            if (allSameRank) {
                uint64_t transferCards = hand & bb::rankMask(atkRank);
                int pairsOnTable = bb::popcount(tableAtk);
                if (bb::popcount(transferCards) >= pairsOnTable &&
                    oppCardCount >= pairsOnTable + bb::popcount(transferCards)) {
                    uint64_t tc = transferCards;
                    while (tc) {
                        int ci = bb::popLowest(tc);
                        moves.push_back(encodeMove(ci, MOVE_TRANSFER));
                    }
                }
            }
        }
    } else {
        if (tableAtk != 0) {
            uint64_t tableRanks = 0;
            uint64_t tTmp = tableAtk | tableDef;
            while (tTmp) {
                int ti = bb::popLowest(tTmp);
                tableRanks |= bb::rankMask(bb::rankOf(ti));
            }
            uint64_t canThrow = hand & tableRanks;
            int maxPairs = pairsLimit;
            if (firstTrick) maxPairs = std::min(maxPairs, 5);
            int currentPairs = bb::popcount(tableAtk);
            if (currentPairs < maxPairs) {
                uint64_t ct = canThrow;
                while (ct) {
                    int ci = bb::popLowest(ct);
                    moves.push_back(encodeMove(ci, MOVE_ATTACK));
                }
            }
        }
        if (tableAtk != 0 && (tableAtk & ~tableDef) == 0) {
            moves.push_back(encodeMove(0, MOVE_BITO));
        }
    }

    return moves;
}

float MctsEngine::gpuRolloutBatch(
    uint64_t hand, uint64_t oppHand,
    uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trump, int phase,
    int attacker, int botPlayer, int pairsLimit,
    bool firstTrick, int count
) {
    if (!gpu_ || !gpu_->isReady() || count <= 0) return 0.5f;

    std::vector<GpuGameState> states(count);
    for (int i = 0; i < count; ++i) {
        memset(&states[i], 0, sizeof(GpuGameState));
        states[i].hand[botPlayer] = hand;
        states[i].hand[1 - botPlayer] = oppHand;
        states[i].table_atk = tableAtk;
        states[i].table_def = tableDef;
        states[i].deck = deck;
        states[i].trump_suit = (uint8_t)trump;
        states[i].phase = (uint8_t)phase;
        states[i].attacker = (uint8_t)attacker;
        states[i].deck_count = (uint8_t)bb::popcount(deck);
        states[i].table_pairs = (uint8_t)bb::popcount(tableAtk);
        states[i].pairs_limit = (uint8_t)pairsLimit;
        states[i].first_trick = firstTrick ? 1 : 0;
    }

    auto result = gpu_->runRollouts(states, botPlayer);
    return result.winRate;
}

void MctsEngine::computePriors(
    int nodeIdx, uint64_t hand, uint64_t oppHand,
    uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trump, int phase, int attacker,
    int botPlayer, int pairsLimit, bool firstTrick
) {
    if (!gpu_ || !gpu_->isReady()) return;

    std::vector<int> children;
    for (int c = tree_[nodeIdx].firstChild; c >= 0; c = tree_[c].nextSibling) {
        children.push_back(c);
    }
    if (children.empty()) return;

    std::vector<GpuGameState> states(children.size());
    for (size_t i = 0; i < children.size(); ++i) {
        memset(&states[i], 0, sizeof(GpuGameState));
        states[i].hand[botPlayer] = hand;
        states[i].hand[1 - botPlayer] = oppHand;
        states[i].table_atk = tableAtk;
        states[i].table_def = tableDef;
        states[i].deck = deck;
        states[i].trump_suit = (uint8_t)trump;
        states[i].phase = (uint8_t)phase;
        states[i].attacker = (uint8_t)attacker;
        states[i].deck_count = (uint8_t)bb::popcount(deck);
        states[i].table_pairs = (uint8_t)bb::popcount(tableAtk);
        states[i].pairs_limit = (uint8_t)pairsLimit;
        states[i].first_trick = firstTrick ? 1 : 0;

        int m = tree_[children[i]].move;
        int card = moveCard(m);
        int act = moveAction(m);
        if (act == MOVE_ATTACK || act == MOVE_PURSUE) {
            states[i].hand[botPlayer] &= ~(1ULL << card);
            states[i].table_atk |= (1ULL << card);
        } else if (act == MOVE_DEFEND) {
            states[i].hand[botPlayer] &= ~(1ULL << card);
            states[i].table_def |= (1ULL << card);
        }
    }

    auto scores = gpu_->evaluateBatch(states, botPlayer);

    float maxS = -1e9f;
    for (float s : scores) if (s > maxS) maxS = s;

    float sumExp = 0.0f;
    std::vector<float> priors(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        priors[i] = expf(scores[i] - maxS);
        sumExp += priors[i];
    }

    for (size_t i = 0; i < children.size(); ++i) {
        tree_[children[i]].prior = (sumExp > 0.0f) ? priors[i] / sumExp : (1.0f / children.size());
    }
}