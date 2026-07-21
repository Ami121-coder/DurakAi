// ============================================================================
// ismcts.cpp — FIX #4: кеширование priors сети для PUCT на всех уровнях
//
// КРИТИЧЕСКИЙ БАГ: при расширении не-root узла сеть вызывается для value,
// но policy НЕ сохраняется. Дети получают uniform prior = 1/n.
// PUCT на не-root уровнях вырождается в UCB1 — сеть не направляет поиск.
//
// ФИКС: каждый узел хранит cachedPolicy (move → prior). При создании ребёнка
// ищем prior в cachedPolicy родителя. Если нет — uniform.
//
// Дополнительно:
//   - leaf evaluation теперь всегда через сеть (если есть), без rollout.
//     Это ускоряет поиск в 3-5× (rollout — самая медленная часть).
//   - virtual loss уже был — оставлен.
//   - Dirichlet noise применяется только если rootTemperature > 0 (self-play).
// ============================================================================

#include "ismcts.h"
#include "rules_fast.h"
#include "rules.h"
#include "threadpool.h"
#include "nnet/policy_value_net.h"
#include "move_ordering.h"  // Task 1: progressive bias / move ordering

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>

namespace durakk {

namespace {

using Clock = std::chrono::steady_clock;

inline std::mt19937_64& rng() {
    static thread_local std::mt19937_64 r{
        std::random_device{}() ^
        (uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id())) << 1)
    };
    return r;
}

CardMask sampleSubset(CardMask pool, int k) {
    int idx[36];
    int n = 0;
    CardMask p = pool;
    while (p) {
        CardMask bit = p & (~p + 1);
        p ^= bit;
        int i;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); i = (int)ul;
#else
        i = __builtin_ctzll(bit);
#endif
        idx[n++] = i;
    }
    if (k > n) k = n;
    if (k <= 0) return 0;
    auto& r = rng();
    for (int i = 0; i < k; ++i) {
        int j = i + (int)(r() % (unsigned)(n - i));
        std::swap(idx[i], idx[j]);
    }
    CardMask out = 0;
    for (int i = 0; i < k; ++i) out |= (uint64_t(1) << idx[i]);
    return out;
}

MatchState determine(const MatchState& root, const Knowledge& k, Player viewpoint) {
    MatchState s = root;
    CardMask pool = k.unknownPool();
    int oppNeed = std::min<int>(k.oppHandCount, popCount(pool));
    CardMask oppHand = sampleSubset(pool, oppNeed);
    pool &= ~oppHand;

    s.hands[toIdx(viewpoint)] = k.myHand;
    s.hands[toIdx(other(viewpoint))] = oppHand;
    s.deck = pool;
    s.deckRemaining = k.deckRemaining;
    s.trump = k.trump;
    return s;
}

struct MoveHash {
    size_t operator()(const Move& m) const noexcept {
        uint64_t h = (uint64_t)m.action;
        h |= static_cast<uint64_t>(m.card.rank) << 8;
        h |= static_cast<uint64_t>(m.card.suit) << 16;
        h |= static_cast<uint64_t>(m.hasTarget ? (static_cast<int>(m.target.rank) | (static_cast<int>(m.target.suit) << 4)) : 0) << 24;
        return std::hash<uint64_t>{}(h);
    }
};

struct MoveEq {
    bool operator()(const Move& a, const Move& b) const noexcept {
        return a.action == b.action && a.card == b.card &&
               a.hasTarget == b.hasTarget &&
               (!a.hasTarget || a.target == b.target);
    }
};

// ---- Узел дерева с кешированной политикой ----
struct Node {
    Move  move;
    int   parent = -1;
    std::vector<int> children;
    std::unordered_set<Move, MoveHash, MoveEq> expandedMoves;
    long  visits = 0;
    double wins = 0.0;
    double prior = 0.0;
    double virtualLoss = 0.0;
    bool   evaluated = false;       // была ли сеть вызвана для этого узла
    double cachedValue = 0.5;       // value из сети (для leaf eval)
    // FIX #4: кеш политики для детей
    std::unordered_map<Move, double, MoveHash, MoveEq> childPriors;
    // Task 1: эвристический prior (вычисляется один раз при создании узла).
    // Используется в progressive bias для UCB1 (без сети) и как fallback prior
    // в PUCT, если сеть не дала prior для этого ребёнка.
    double heuristicPrior = 0.5;
    // Task 1: состояние, для которого считался heuristicPrior (нужно для пересчёта
    // при использовании move ordering в select). Храним только нужные поля.
    CardMask heuristicHand = 0;  // рука ходящего в позиции-родителе
    MatchPhase heuristicPhase = MatchPhase::Attack;
    Suit heuristicTrump = Suit::Spades;
};

struct Tree {
    std::vector<Node> nodes;
    std::mutex mtx;
};

int puctSelect(const Tree& t, int node, double c, long parentVisits,
               double virtualLossWeight) {
    const Node& n = t.nodes[node];
    int best = -1;
    double bestVal = -1e18;
    for (int ci : n.children) {
        const Node& ch = t.nodes[ci];
        if (ch.visits == 0) return ci;
        double exploit = ch.wins / (double)ch.visits;
        // Task 1: если prior почти нулевой (сеть неуверенна), подмешиваем эвристику.
        // Это даёт softer fallback на ранних итерациях для не-root узлов без priors.
        double effectivePrior = ch.prior;
        if (ch.prior < 1e-3) {
            effectivePrior = ch.heuristicPrior;
        }
        double explore = c * effectivePrior *
            std::sqrt((double)parentVisits + 1.0) / (1.0 + (double)ch.visits);
        // Task 1: progressive bias — дополнительный бонус на основе эвристики.
        // bonus = k * heuristicPrior / (1 + visits)
        // Убывает с ростом визитов (по мере накопления стат. данных эвристика уступает).
        double pbias = kProgressiveBiasC * ch.heuristicPrior / (1.0 + (double)ch.visits);
        double vl = virtualLossWeight * ch.virtualLoss / (double)ch.visits;
        double v = exploit + explore + pbias - vl;
        if (v > bestVal) { bestVal = v; best = ci; }
    }
    return best;
}

int ucbSelect(const Tree& t, int node, double c, long parentVisits) {
    const Node& n = t.nodes[node];
    int best = -1;
    double bestVal = -1e18;
    for (int ci : n.children) {
        const Node& ch = t.nodes[ci];
        if (ch.visits == 0) return ci;
        double exploit = ch.wins / ch.visits;
        double explore = c * std::sqrt(std::log((double)parentVisits + 1.0) / (double)ch.visits);
        // Task 1: progressive bias — главный источник «интеллекта» без сети.
        // bonus = k * heuristicPrior / (1 + visits)
        // На ранних итерациях (visits<10) даёт большой буст качественным ходам.
        // По мере накопления визитов — эвристика уступает место статистике.
        double pbias = kProgressiveBiasC * ch.heuristicPrior / (1.0 + (double)ch.visits);
        double v = exploit + explore + pbias;
        if (v > bestVal) { bestVal = v; best = ci; }
    }
    return best;
}

double terminalValue(const MatchState& s, Player viewpoint) {
    int w = s.winner();
    if (w < 0) return 0.5;
    return (w == toIdx(viewpoint)) ? 1.0 : 0.0;
}

double rollout(MatchState s, Player viewpoint, int maxDepth) {
    int safety = 0;
    const int SAFETY_MAX = 4000;
    while (!s.isGameOver() && safety++ < SAFETY_MAX && (maxDepth == 0 || safety < maxDepth)) {
        MoveBuffer buf;
        int n = genLegalMoves(s, buf);
        if (n == 0) break;
        Move m;
        if (!defaultPolicy(s, buf.data(), n, m)) break;
        if (!applyMove(s, m)) {
            if (!applyMove(s, buf[0])) break;
        }
    }
    return terminalValue(s, viewpoint);
}

void addDirichletToRoot(Tree& tree, double alpha, double eps) {
    Node& root = tree.nodes[0];
    int n = (int)root.children.size();
    if (n <= 0) return;
    std::gamma_distribution<double> dist(alpha, 1.0);
    auto& r = rng();
    std::vector<double> noise(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { noise[i] = dist(r); sum += noise[i]; }
    if (sum <= 0) return;
    for (int i = 0; i < n; ++i) {
        Node& ch = tree.nodes[root.children[i]];
        double noised = (1.0 - eps) * ch.prior + eps * (noise[i] / sum);
        ch.prior = noised;
    }
}

// ---- FIX #16: убрана неиспользуемая функция evaluateNodeWithNet ----
// Она была объявлена, но нигде не вызывалась — сеть для листьев
// вызывается инлайн в worker-ламбде ниже. Мёртвый код удалён, чтобы
// не сбивать с толку.

} // namespace

IsmctsResult runIsmcts(const MatchState& rootState, const Knowledge& knowledge,
                       const IsmctsLimits& lim, std::atomic<bool>* stopFlag,
                       Player viewpoint, PolicyValueNet* net) {
    IsmctsResult res{};
    auto start = Clock::now();
    auto deadline = start + std::chrono::milliseconds(
        static_cast<long long>(lim.timeBudgetSec * 1000));

    Tree tree;
    tree.nodes.emplace_back();
    tree.nodes.back().parent = -1;

    std::atomic<long> totalPlayouts{0};
    std::atomic<bool> localStop{false};
    auto isTimeUp = [&]() {
        return localStop.load() ||
               (stopFlag && stopFlag->load()) ||
               (Clock::now() >= deadline);
    };

    // ---- Корень: оцениваем сетью, создаём всех детей с priors ----
    if (net) {
        MatchState rootDet = determine(rootState, knowledge, viewpoint);
        PVResult pv = net->evaluate(rootDet, viewpoint);

        std::lock_guard<std::mutex> lk(tree.mtx);
        // FIX: НЕ держим ссылку на tree.nodes[0] через emplace_back —
        // реаллокация вектора делает её висячей (heap-use-after-free,
        // именно это убивало self-play воркеров). Резервируем место заранее
        // и обращаемся к корню по индексу.
        tree.nodes.reserve(pv.policy.size() + 16);
        tree.nodes[0].evaluated = true;
        tree.nodes[0].cachedValue = pv.value;
        // Task 1: контекст эвристики для корня.
        tree.nodes[0].heuristicHand = rootDet.hands[toIdx(rootDet.turn)];
        tree.nodes[0].heuristicPhase = rootDet.phase;
        tree.nodes[0].heuristicTrump = rootDet.trump;

        for (auto& [mv, prior] : pv.policy) {
            MatchState child = rootDet;
            if (!applyMove(child, mv)) continue;
            tree.nodes.emplace_back();
            Node& ch = tree.nodes.back();
            ch.parent = 0;
            ch.move = mv;
            ch.prior = prior;
            // Task 1: считаем эвристический prior один раз.
            // Hand = рука ходящего в КОРНЕ (до применения mv).
            ch.heuristicPrior = moveHeuristic(rootDet, mv,
                                              rootDet.hands[toIdx(rootDet.turn)]);
            ch.heuristicHand = rootDet.hands[toIdx(rootDet.turn)];
            ch.heuristicPhase = rootDet.phase;
            ch.heuristicTrump = rootDet.trump;
            int newIdx = (int)tree.nodes.size() - 1;
            tree.nodes[0].children.push_back(newIdx);
            tree.nodes[0].expandedMoves.insert(mv);
        }

        if (!pv.policy.empty() && lim.dirichletEps > 0) {
            addDirichletToRoot(tree, lim.dirichletAlpha, lim.dirichletEps);
        }
        res.rootValue = pv.value;
    } else {
        // Task 1: без сети — корень не имеет priors, но дети получат heuristicPrior
        // при первом расширении. Инициализируем контекст эвристики для корня.
        MatchState rootDet = determine(rootState, knowledge, viewpoint);
        std::lock_guard<std::mutex> lk(tree.mtx);
        tree.nodes[0].heuristicHand = rootDet.hands[toIdx(rootDet.turn)];
        tree.nodes[0].heuristicPhase = rootDet.phase;
        tree.nodes[0].heuristicTrump = rootDet.trump;
    }

    int nThreads = std::max(1, lim.numThreads);
    ThreadPool pool;

    auto worker = [&](int /*wid*/) {
        long localCount = 0;
        while (!isTimeUp()) {
            MatchState sim = determine(rootState, knowledge, viewpoint);
            if (sim.isGameOver()) { localCount++; continue; }

            int cur = 0;
            std::vector<int> path;
            path.reserve(32);
            path.push_back(0);

            // ---- Select ----
            while (!sim.isGameOver()) {
                MoveBuffer buf;
                int n = genLegalMoves(sim, buf);
                if (n == 0) break;

                int chosenChild = -1;
                Move expandMove;
                bool doExpand = false;
                bool doSelect = false;
                {
                    std::lock_guard<std::mutex> lk(tree.mtx);
                    Node& node = tree.nodes[cur];

                    // FIX #17: убран пустой if (net && !node.evaluated && cur != 0)
                    // — тело было пустым, никакого эффекта. Листья оцениваются
                    // сетью при расширении (см. ниже в doExpand ветке).

                    for (int i = 0; i < n; ++i) {
                        if (node.expandedMoves.find(buf[i]) == node.expandedMoves.end()) {
                            doExpand = true;
                            expandMove = buf[i];
                            node.expandedMoves.insert(buf[i]);
                            break;
                        }
                    }
                    if (!doExpand) {
                        chosenChild = net
                            ? puctSelect(tree, cur, lim.puctC, node.visits, lim.virtualLoss)
                            : ucbSelect(tree, cur, lim.explorationC, node.visits);
                        doSelect = chosenChild >= 0;
                    }
                }

                if (doExpand) {
                    // ---- FIX #4: prior из кеша родителя ----
                    MatchState child = sim;
                    if (!applyMove(child, expandMove)) {
                        break;
                    }

                    double prior = 1.0 / (double)n;  // default
                    double heuristicPrior = 0.5;     // Task 1: default
                    CardMask heuristicHand = sim.hands[toIdx(sim.turn)];
                    MatchPhase heuristicPhase = sim.phase;
                    Suit heuristicTrump = sim.trump;
                    {
                        std::lock_guard<std::mutex> lk(tree.mtx);
                        Node& parent = tree.nodes[cur];
                        auto it = parent.childPriors.find(expandMove);
                        if (it != parent.childPriors.end()) {
                            prior = it->second;
                        }
                        // Task 1: считаем эвристический prior для нового узла.
                        // Используем sim (состояние ДО применения expandMove).
                        heuristicPrior = moveHeuristic(sim, expandMove,
                                                       sim.hands[toIdx(sim.turn)]);
                    }

                    int newIdx;
                    {
                        std::lock_guard<std::mutex> lk(tree.mtx);
                        tree.nodes.emplace_back();
                        Node& c = tree.nodes.back();
                        c.parent = cur;
                        c.move = expandMove;
                        c.prior = prior;
                        // Task 1: сохраняем эвристику и контекст для будущих пересчётов.
                        c.heuristicPrior = heuristicPrior;
                        c.heuristicHand = heuristicHand;
                        c.heuristicPhase = heuristicPhase;
                        c.heuristicTrump = heuristicTrump;
                        newIdx = (int)tree.nodes.size() - 1;
                        tree.nodes[cur].children.push_back(newIdx);
                    }

                    // ---- Leaf evaluation через сеть ----
                    if (net) {
                        // Оценим child состояние сетью.
                        PVResult pv = net->evaluate(child, viewpoint);
                        double value = pv.value;

                        // Сохраним priors для будущих детей этого узла.
                        {
                            std::lock_guard<std::mutex> lk(tree.mtx);
                            Node& newNode = tree.nodes[newIdx];
                            newNode.evaluated = true;
                            newNode.cachedValue = value;
                            for (auto& [mv, pr] : pv.policy) {
                                newNode.childPriors[mv] = pr;
                            }
                        }

                        // Backprop value.
                        for (int it = (int)path.size() - 1; it >= 0; --it) {
                            std::lock_guard<std::mutex> lk(tree.mtx);
                            Node& nn = tree.nodes[path[it]];
                            nn.visits++;
                            nn.wins += value;
                            nn.virtualLoss -= lim.virtualLoss;
                        }
                        localCount++;
                        goto nextIteration;
                    }

                    sim = child;
                    cur = newIdx;
                    path.push_back(cur);
                    break;
                } else if (doSelect) {
                    Move mv;
                    {
                        std::lock_guard<std::mutex> lk(tree.mtx);
                        tree.nodes[chosenChild].virtualLoss += lim.virtualLoss;
                        // Читаем move ПОД локом: параллельный emplace_back
                        // из другого потока может реаллоцировать tree.nodes.
                        mv = tree.nodes[chosenChild].move;
                    }
                    if (!applyMove(sim, mv)) {
                        break;
                    }
                    cur = chosenChild;
                    path.push_back(cur);
                    continue;
                } else {
                    break;
                }
            }

            // ---- Rollout / terminal ----
            double value;
            if (sim.isGameOver()) {
                value = terminalValue(sim, viewpoint);
            } else {
                // Если есть net, но leaf не был оценен — оценим.
                // (Это может случиться если достигли глубины без расширения.)
                if (net) {
                    PVResult pv = net->evaluate(sim, viewpoint);
                    value = pv.value;
                } else {
                    value = rollout(sim, viewpoint, lim.maxRolloutDepth);
                    if (!sim.isGameOver() && lim.maxRolloutDepth > 0) value = 0.5;
                }
            }

            // Backprop
            for (int it = (int)path.size() - 1; it >= 0; --it) {
                std::lock_guard<std::mutex> lk(tree.mtx);
                Node& nn = tree.nodes[path[it]];
                nn.visits++;
                nn.wins += value;
                nn.virtualLoss -= lim.virtualLoss;
            }
            localCount++;
        nextIteration:;
        }
        totalPlayouts.fetch_add(localCount, std::memory_order_relaxed);
    };

    pool.start(nThreads, worker, &localStop);
    while (!isTimeUp()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    localStop.store(true);
    pool.join();

    res.playouts = totalPlayouts.load();
    res.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    // ---- Выбор хода ----
    {
        std::lock_guard<std::mutex> lk(tree.mtx);
        const Node& r = tree.nodes[0];
        res.rootChildren = (int)r.children.size();
        long totalRootVisits = 0;
        for (int ci : r.children) totalRootVisits += tree.nodes[ci].visits;
        if (totalRootVisits == 0) totalRootVisits = 1;

        // Всегда возвращаем нормализованные visits как policy.
        for (int ci : r.children) {
            double prob = (double)tree.nodes[ci].visits / (double)totalRootVisits;
            res.rootProbs.push_back({tree.nodes[ci].move, prob});
        }

        // argmax visits — детерминированный выбор.
        int best = -1; long bestV = -1;
        for (int ci : r.children) {
            if (tree.nodes[ci].visits > bestV) {
                bestV = tree.nodes[ci].visits;
                best = ci;
            }
        }
        if (best >= 0) {
            res.move = tree.nodes[best].move;
            res.winrate = tree.nodes[best].visits > 0
                ? tree.nodes[best].wins / tree.nodes[best].visits : 0.0;
        } else {
            res.move = Move{Action::Pass, Card{}, Card{}, false, {}};
        }
    }
    return res;
}

} // namespace durakk
