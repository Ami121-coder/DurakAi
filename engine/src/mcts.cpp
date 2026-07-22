#include "mcts.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <random>

MctsEngine::MctsEngine() {}
MctsEngine::~MctsEngine() {}

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
    int oppCardCount, const float bayesWeights[36]
) {
    auto t0 = std::chrono::steady_clock::now();

    tree_.clear();
    tree_.reserve(cfg_.maxTreeNodes);

    // Корневой узел
    MctsNode root{};
    root.player = (uint8_t)botPlayer;
    tree_.push_back(root);

    int totalIter = 0;
    int totalGpuRollouts = 0;

    // Если рука соперника неизвестна — будем семплировать
    bool hiddenOpp = (oppHand == 0 && oppCardCount > 0);

    while (totalIter < cfg_.maxIterations) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        if (elapsed >= cfg_.timeLimitMs) break;

        // Семплируем руку соперника если скрыта
        uint64_t curOppHand = oppHand;
        if (hiddenOpp) {
            uint64_t known = botHand | tableAtk | tableDef;
            // Простое семплирование на CPU (GPU-семплинг для больших батчей)
            uint64_t pool = ((1ULL << 36) - 1) & ~known;
            curOppHand = 0;
            std::mt19937 rng(totalIter + 42);
            std::vector<int> indices;
            for (int i = 0; i < 36; ++i)
                if (pool & (1ULL << i)) indices.push_back(i);
            // Weighted shuffle
            for (int i = (int)indices.size() - 1; i > 0; --i) {
                std::uniform_real_distribution<float> dist(0, 1);
                float total = 0;
                for (int j = 0; j <= i; ++j) total += bayesWeights[indices[j]];
                float r = dist(rng) * total;
                float acc = 0;
                for (int j = 0; j <= i; ++j) {
                    acc += bayesWeights[indices[j]];
                    if (acc >= r) { std::swap(indices[i], indices[j]); break; }
                }
            }
            for (int i = 0; i < oppCardCount && i < (int)indices.size(); ++i)
                curOppHand |= (1ULL << indices[i]);
        }

        // 1. Selection
        int node = selectNode(0);

        // 2. Expansion
        uint64_t h = botHand, oh = curOppHand;
        uint64_t ta = tableAtk, td = tableDef, dk = deck;
        int ph = phase, atk = attacker, dkCnt = bb::popcount(deck);

        // Восстанавливаем состояние по пути (упрощённо: для глубины 1)
        if (node > 0) {
            // Для простоты: expansion только из корня на первых итерациях
            // Полная реализация: хранить состояние в каждом узле
        }

        int expanded = expandNode(node, h, oh, ta, td, dk, trumpSuit, ph, atk);

        // 3. Simulation (GPU rollout)
        float value;
        if (expanded >= 0 && expanded < (int)tree_.size()) {
            value = gpuRolloutBatch(h, oh, ta, td, dk, trumpSuit, ph, atk, botPlayer,
                                    cfg_.gpuRolloutsPerLeaf);
            totalGpuRollouts += cfg_.gpuRolloutsPerLeaf;
        } else {
            value = 0.5f;
        }

        // 4. Backpropagation
        backpropagate(node, value);
        totalIter++;

        // Периодический GPU-батч для ускорения
        if (totalIter % 1000 == 0 && gpu_ && gpu_->isReady()) {
            // Собираем листья и делаем большой батч
            // (реализация ниже в gpuRolloutBatch)
        }
    }

    // Выбираем лучший ход из корня
    int bestChild = -1;
    int bestVisits = -1;
    for (int c = tree_[0].firstChild; c >= 0; c = tree_[c].nextSibling) {
        if (tree_[c].visits > bestVisits) {
            bestVisits = tree_[c].visits;
            bestChild = c;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    lastStats_.totalIterations = totalIter;
    lastStats_.gpuRollouts = totalGpuRollouts;
    lastStats_.treeNodes = (int)tree_.size();
    lastStats_.timeMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    lastStats_.bestWinRate = (bestChild >= 0) ?
        tree_[bestChild].totalValue / std::max(1, tree_[bestChild].visits) : 0.5f;

    return (bestChild >= 0) ? tree_[bestChild].move : -1;
}

int MctsEngine::selectNode(int nodeIdx) {
    int cur = nodeIdx;
    while (cur >= 0 && cur < (int)tree_.size()) {
        if (tree_[cur].terminal) return cur;
        if (tree_[cur].untriedMoves > 0) return cur;
        if (tree_[cur].firstChild < 0) return cur;

        // UCB1 selection
        float bestUcb = -1e9f;
        int bestChild = -1;
        float parentVisits = (float)std::max(1, tree_[cur].visits);

        for (int c = tree_[cur].firstChild; c >= 0; c = tree_[c].nextSibling) {
            float childVisits = (float)std::max(1, tree_[c].visits);
            float exploit = tree_[c].totalValue / childVisits;
            float explore = cfg_.ucbC * sqrtf(logf(parentVisits) / childVisits);
            float ucb = exploit + explore;
            if (ucb > bestUcb) {
                bestUcb = ucb;
                bestChild = c;
            }
        }
        cur = bestChild;
    }
    return cur;
}

int MctsEngine::expandNode(int nodeIdx, uint64_t hand, uint64_t oppHand,
                           uint64_t tableAtk, uint64_t tableDef,
                           uint64_t deck, int trump, int phase, int attacker) {
    if (nodeIdx < 0 || nodeIdx >= (int)tree_.size()) return -1;
    if (tree_[nodeIdx].terminal) return nodeIdx;

    int player = tree_[nodeIdx].player;
    uint64_t myHand = (player == 0) ? hand : oppHand;

    auto moves = genMoves(myHand, tableAtk, tableDef, deck, trump, phase, attacker,
                          player, bb::popcount((player == 0) ? oppHand : hand));

    if (moves.empty()) {
        tree_[nodeIdx].terminal = true;
        tree_[nodeIdx].terminalValue = 0.5f;
        return nodeIdx;
    }

    tree_[nodeIdx].untriedMoves = (int)moves.size();

    // Создаём дочерние узлы
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

    tree_[nodeIdx].untriedMoves = 0;
    return tree_[nodeIdx].firstChild;
}

float MctsEngine::gpuRolloutBatch(
    uint64_t hand, uint64_t oppHand,
    uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trump, int phase,
    int attacker, int botPlayer, int count
) {
    if (!gpu_ || !gpu_->isReady() || count <= 0) return 0.5f;

    // Заполняем батч состояний
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
    }

    auto result = gpu_->runRollouts(states, botPlayer);
    return result.winRate;
}

void MctsEngine::backpropagate(int nodeIdx, float value) {
    int cur = nodeIdx;
    while (cur >= 0) {
        tree_[cur].visits++;
        // Инвертируем value для каждого уровня (zero-sum)
        tree_[cur].totalValue += (tree_[cur].player == 0) ? value : (1.0f - value);
        cur = tree_[cur].parent;
    }
}

std::vector<int> MctsEngine::genMoves(
    uint64_t hand, uint64_t tableAtk, uint64_t tableDef,
    uint64_t deck, int trump, int phase, int attacker,
    int player, int oppCardCount
) {
    std::vector<int> moves;
    if (hand == 0) return moves;

    uint64_t trumpBits = bb::trumpMask(trump);

    if (phase == 0 && player == attacker) {
        // АТАКА: можно положить любую карту (или "бито" если стол пуст)
        uint64_t tmp = hand;
        while (tmp) {
            int idx = bb::popLowest(tmp);
            moves.push_back(encodeMove(idx, MOVE_ATTACK));
        }
        // Если стол не пуст и всё побито — можно "бито"
        if (tableAtk != 0 && (tableAtk & ~tableDef) == 0) {
            moves.push_back(encodeMove(0, MOVE_BITO));
        }
    } else if (phase == 1 && player != attacker) {
        // ЗАЩИТА: бьём или берём
        uint64_t undefended = tableAtk & ~tableDef;
        if (undefended == 0) {
            // Всё побито — "бито"
            moves.push_back(encodeMove(0, MOVE_BITO));
            return moves;
        }

        // Для каждой непокрытой атаки — ищем чем побить
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
                // Не можем побить эту карту — "беру"
                moves.push_back(encodeMove(0, MOVE_TAKE));
                return moves; // "беру" — единственный вариант
            }
            break; // обрабатываем первую непокрытую
        }

        // Перевод: если все карты на столе одного ранга и у нас есть такая же
        // + у соперника достаточно карт
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
                    // Можно перевести
                    uint64_t tc = transferCards;
                    while (tc) {
                        int ci = bb::popLowest(tc);
                        moves.push_back(encodeMove(ci, MOVE_TRANSFER));
                    }
                }
            }
        }
    } else {
        // Подкидывание (атакующий после защиты)
        if (tableAtk != 0) {
            // Можно подкинуть карты тех рангов, что на столе
            uint64_t tableRanks = 0;
            uint64_t tTmp = tableAtk | tableDef;
            while (tTmp) {
                int ti = bb::popLowest(tTmp);
                tableRanks |= bb::rankMask(bb::rankOf(ti));
            }
            uint64_t canThrow = hand & tableRanks;
            int maxPairs = 6;
            if (bb::popcount(deck) == 0) maxPairs = 5; // последняя рука
            int currentPairs = bb::popcount(tableAtk);
            if (currentPairs < maxPairs) {
                uint64_t ct = canThrow;
                while (ct) {
                    int ci = bb::popLowest(ct);
                    moves.push_back(encodeMove(ci, MOVE_ATTACK));
                }
            }
        }
        // "Бито"
        if (tableAtk != 0 && (tableAtk & ~tableDef) == 0) {
            moves.push_back(encodeMove(0, MOVE_BITO));
        }
    }

    return moves;
}

void MctsEngine::applyMove(int move, uint64_t& hand, uint64_t& oppHand,
                           uint64_t& tableAtk, uint64_t& tableDef,
                           uint64_t& deck, int& phase, int& attacker, int& deckCnt) {
    int card = moveCard(move);
    int action = moveAction(move);

    switch (action) {
    case MOVE_ATTACK:
        hand &= ~(1ULL << card);
        tableAtk |= (1ULL << card);
        phase = 1;
        break;
    case MOVE_DEFEND:
        hand &= ~(1ULL << card);
        tableDef |= (1ULL << card);
        if ((tableAtk & ~tableDef) == 0) {
            // Всё побито
            tableAtk = 0; tableDef = 0;
            phase = 0;
            attacker = 1 - attacker;
        }
        break;
    case MOVE_TAKE:
        hand |= tableAtk | tableDef;
        tableAtk = 0; tableDef = 0;
        phase = 0;
        attacker = 1 - attacker;
        break;
    case MOVE_BITO:
        tableAtk = 0; tableDef = 0;
        phase = 0;
        attacker = 1 - attacker;
        break;
    case MOVE_TRANSFER:
        hand &= ~(1ULL << card);
        tableAtk |= (1ULL << card);
        // Перевод: атакующий меняется
        attacker = 1 - attacker;
        phase = 1;
        break;
    }
}
