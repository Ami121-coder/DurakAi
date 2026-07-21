#include "bot.h"

#include "endgame.h"
#include "endgame_db.h"  // Task 7
#include "ismcts.h"
#include "move_ordering.h"  // Task 6: orderMoves для pondering
#include "rules_fast.h"
#include "rules.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>

namespace durakk {

namespace {

// Локальный weighted sampling руки соперника для эндшпиля.
// Когда колода пуста (deckRemaining==0), все карты unknownPool физически
// находятся у соперника — но из-за возможных ошибок трекинга Knowledge
// popCount(unknownPool) может не совпадать с k.oppHandCount. Поэтому:
//   • если совпадает точно — берём все (идеальный случай);
//   • если неизвестных больше, чем нужно — сэмплируем oppHandCount штук;
//   • если меньше — берём сколько есть.
CardMask sampleOppHandForEndgame(CardMask pool, int need) {
    int n = popCount(pool);
    if (n == 0 || need <= 0) return 0;
    if (need >= n) return pool;  // все неизвестные = у соперника

    int idx[36];
    int cnt = 0;
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
        idx[cnt++] = i;
    }
    static thread_local std::mt19937_64 rng{ std::random_device{}() };
    for (int i = 0; i < need; ++i) {
        int j = i + (int)(rng() % (unsigned)(cnt - i));
        std::swap(idx[i], idx[j]);
    }
    CardMask out = 0;
    for (int i = 0; i < need; ++i) out |= (uint64_t(1) << idx[i]);
    return out;
}

// Параметры силы → таймаут.
double timeoutFor(Strength s) {
    switch (s) {
        case Strength::Fast:   return 0.5;
        case Strength::Normal: return 2.0;
        case Strength::Deep:   return 10.0;
    }
    return 2.0;
}

// Построить MatchState из GameState (наблюдаемой позиции).
// Рука соперника и колода здесь — заглушки (0); реальное их содержание добавляет
// ISMCTS через determine() в каждой симуляции.
MatchState toMatchState(const GameState& s, const Knowledge& k) {
    MatchState m{};
    m.trump = s.deck.trump;
    m.hands[0] = k.myHand;       // рука бота
    m.hands[1] = 0;              // соперник — неизвестен (ISMCTS раздасm)
    m.deck = 0;
    m.deckRemaining = s.deck.remaining;
    m.firstTrick = s.firstTrick;
    m.transferEnabled = s.transferEnabled;
    m.flashEnabled = s.flashEnabled;
    // FIX #18: пробрасываем пользовательскую настройку лимита первого кона.
    // Раньше хардкодилось pairsLimit = 6, а firstTrick ? 5 тоже хардкодилось
    // в pairsHeadroom() — это расходилось с rules.cpp::maxPairsThisTrick,
    // который использовал firstTrickLimit из GameState.
    m.pairsLimit = s.firstTrick ? std::min(s.firstTrickLimit, 5)
                                : 6;
    m.attacker = (s.attacker == Side::Me) ? Player::Me : Player::Opp;
    m.turn = (s.turn == Side::Me) ? Player::Me : Player::Opp;
    m.phase = (s.phase == Phase::Attack) ? MatchPhase::Attack : MatchPhase::Defense;

    // Стол: скопируем пары. Карта-на-руке бота из Knowledge уже учтена в m.hands[0].
    for (const auto& p : s.table) {
        if (m.tableLen >= 6) break;
        Pair& tp = m.table[m.tableLen++];
        tp.attack = p.attack;
        if (p.defended) {
            tp.defense = p.defense;
            tp.defended = true;
        }
    }
    return m;
}

} // namespace

Bot::Bot() {
    // По умолчанию бот — чисто математический (без нейросети).
    // Сеть подгружается ОПЦИОНАЛЬНО: если есть checkpoints/model.onnx и она
    // успешно инициализируется — используем её в ISMCTS. Иначе ISMCTS работает
    // по UCB1 + rollout (передаём nullptr), что даёт корректный мат. поиск
    // вместо деградации на RandomNet (uniform priors + value=0.5).
#ifdef DURAKK_USE_ONNX
    try {
        auto onnx = std::make_unique<OnnxNet>("checkpoints/model.onnx", "CUDA", 0);
        if (onnx->isReady()) {
            net_ = std::move(onnx);
            hasRealNet_ = true;
        } else {
            std::fprintf(stderr, "[Bot] OnnxNet не готов — работаем без сети (чистая математика).\n");
        }
    } catch (...) {
        try {
            auto onnx = std::make_unique<OnnxNet>("checkpoints/model.onnx", "CPU", 0);
            if (onnx->isReady()) {
                net_ = std::move(onnx);
                hasRealNet_ = true;
            }
        } catch (...) {
            std::fprintf(stderr, "[Bot] OnnxNet недоступен — работаем без сети (чистая математика).\n");
        }
    }
#endif
}

Move Bot::decide(const GameState& s, const Knowledge& k,
                 const SearchSettings& settings, DecisionStats* statsOut) {
    DecisionStats stats{};
    MatchState root = toMatchState(s, k);

    // ---------- Task 7: Endgame Database O(1) lookup ----------
    // Если загружена БД и позиция подходит (deck=0, стол пуст, суммарно ≤4 карт),
    // возвращаем предрассчитанный ход мгновенно.
    if (endgameDB_.isReady() && s.deck.remaining == 0 && s.table.empty()) {
        int totalCards = static_cast<int>(s.myHand.size()) + s.oppHandCount;
        if (totalCards <= 4) {
            EndgameDBEntry e = endgameDB_.lookup(root);
            if (e.bestCardIdx != 0xFF) {
                // Восстанавливаем Move из entry.
                Move m;
                m.action = static_cast<Action>(e.action);
                if (e.bestCardIdx < 36) {
                    m.card = indexToCard(e.bestCardIdx);
                }
                if (e.targetCardIdx != 0xFF && e.targetCardIdx < 36) {
                    m.target = indexToCard(e.targetCardIdx);
                    m.hasTarget = true;
                }
                m.reason = "EndgameDB: O(1) lookup";

                stats.mode = "EndgameDB";
                stats.timeMs = 0.0;
                stats.solved = (e.result == 1 || e.result == 2);
                if (statsOut) *statsOut = stats;
                return m;
            }
        }
    }

    // ---------- Эндшпиль: колода пуста → minimax α-β (perfect information) ----------
    if (s.deck.remaining == 0) {
        // FIX (этап 0.2): раньше сюда передавался весь unknownPool (мог быть
        // 20+ карт при реальных 3 у соперника) — минимакс играл с нереалистично
        // большой рукой оппонента. Теперь сэмплируем ровно oppHandCount карт.
        // Когда колода пуста, неизвестные карты в норме ВСЕ у соперника
        // (если трекинг Knowledge корректен) — тогда need >= popCount(pool)
        // и sampleOppHandForEndgame вернёт pool целиком без потерь.
        CardMask pool = k.unknownPool();
        root.hands[1] = sampleOppHandForEndgame(pool, k.oppHandCount);
        root.deck = 0;
        root.deckRemaining = 0;

        EndgameLimits lim;
        lim.timeBudgetSec = timeoutFor(settings.strength);
        EndgameResult er = bestEndgameMove(root, Player::Me, lim, nullptr);

        stats.mode = "Endgame";
        stats.depthReached = er.depthReached;
        stats.solved = er.solved;
        stats.timeMs = er.timeMs;
        stats.playouts = er.nodes;
        if (statsOut) *statsOut = stats;
        if (!er.move.reason.empty()) {
            return er.move;
        }
        Move m = er.move;
        m.reason = er.solved
            ? (er.score > 0 ? "эндшпиль: форсированная победа (минимакс)"
                            : "эндшпиль: позиция проиграна (минимакс)")
            : "эндшпиль: лучший ход перебором (α-β)";
        return m;
    }

    // ---------- Task 5: Early Endgame Trigger ----------
    // Когда в колоде осталось мало карт (≤ EARLY_ENDGAME_THRESHOLD), но не 0,
    // ISMCTS тратит бюджет на ветки с высокой вариативностью из-за случайного
    // добора. Sampled minimax даёт более надёжное решение: сэмплируем N
    // детерминизаций руки оппонента (по байесовской матрице oppProbs),
    // для каждой запускаем точный α-β, усредняем winrate.
    constexpr int EARLY_ENDGAME_THRESHOLD = 6;
    if (s.deck.remaining > 0 && s.deck.remaining <= EARLY_ENDGAME_THRESHOLD) {
        EndgameLimits elim;
        elim.timeBudgetSec = timeoutFor(settings.strength);
        // Адаптивное число сэмплов: при budget=2s и ~50ms на сэмпл = ~40 сэмплов.
        // При budget=0.5s = ~10 сэмплов.
        int nSamples = std::max(8, std::min(50,
            (int)(elim.timeBudgetSec * 1000.0 / 50.0)));
        EndgameResult er = bestSampledEndgameMove(root, k, Player::Me,
                                                   elim, nSamples, nullptr);

        if (er.move.action != Action::Pass) {
            stats.mode = "EarlyEndgame";
            stats.depthReached = er.depthReached;
            stats.solved = er.solved;
            stats.timeMs = er.timeMs;
            stats.playouts = er.nodes;
            if (statsOut) *statsOut = stats;

            Move m = er.move;
            if (m.reason.empty()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "Early endgame (deck=%d): %d sampled minimax, score=%d, %.0fms",
                    s.deck.remaining, nSamples, er.score, er.timeMs);
                m.reason = buf;
            }
            return m;
        }
        // Если sampled endgame не дал результат — fallback на ISMCTS ниже.
    }

    // ---------- Фаза с колодой → ISMCTS ----------
    // FIX (этап 0.1): передаём сеть ТОЛЬКО если она реально загружена.
    // При отсутствии сети (по умолчанию для математического бота) — nullptr,
    // тогда ISMCTS идёт по UCB1 + rollout (корректный мат. путь), а не по
    // PUCT с uniform priors из RandomNet (что деградирует поиск).
    IsmctsLimits lim;
    lim.timeBudgetSec = timeoutFor(settings.strength);
    lim.numThreads = settings.numThreads;
    PolicyValueNet* net = hasRealNet_ ? net_.get() : nullptr;
    IsmctsResult r = runIsmcts(root, k, lim, nullptr, Player::Me, net);

    stats.mode = hasRealNet_ ? "ISMCTS+NN" : "ISMCTS";
    stats.playouts = r.playouts;
    stats.winrate = r.winrate;
    stats.timeMs = r.timeMs;
    if (statsOut) *statsOut = stats;

    Move m = r.move;
    if (m.reason.empty()) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "ISMCTS: %ld плейаутов, winrate≈%.0f%%, %d корневых ходов, %.2fs",
            r.playouts, r.winrate * 100.0, r.rootChildren, r.timeMs / 1000.0);
        m.reason = buf;
    }
    return m;
}

// ============================================================================
// Task 6: Pondering — фоновый предрасчёт ходов на ходе соперника.
// ============================================================================

Bot::~Bot() {
    stopPondering();
}

uint64_t Bot::ponderHash(const GameState& s, const Knowledge& k) {
    // Упрощённый хеш: myHand + table + trump + turn + phase.
    // НЕ учитывает deck (он не меняется на ходе соперника, но может
    // отличаться между двумя вызовами из-за добора — это ок для pondering).
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    h ^= k.myHand * 0x100000001B3ULL;
    h ^= k.tableKnown * 0xC2B2AE3D27D4EB4FULL;
    h ^= (uint64_t)s.deck.trump << 16;
    h ^= (uint64_t)s.turn << 24;
    h ^= (uint64_t)s.phase << 28;
    h ^= (uint64_t)s.attacker << 32;
    return h;
}

void Bot::startPondering(const GameState& s, const Knowledge& k,
                         const SearchSettings& settings) {
    // Если уже есть запущенный pondering — остановим.
    stopPondering();

    // Простая проверка: если у обоих 0 карт и стол пуст — игра окончена.
    if (s.myHand.empty() && s.oppHandCount == 0 && s.table.empty()) return;

    ponderStopFlag_.store(false);
    ponderActive_.store(true);

    // Запускаем фоновый поток, который считает ТОП-3 наиболее вероятных
    // ответа соперника (через ISMCTS с viewpoint=Opp) и для каждого
    // предрассчитывает наш лучший ход.
    ponderThread_ = std::thread(&Bot::ponderWorker, this, s, k, settings);
}

void Bot::ponderWorker(const GameState s, const Knowledge k,
                       const SearchSettings settings) {
    // Контекст: тот же матч, но viewpoint = Opp (соперник ходит).
    // Считаем его ходы, для топ-N предсказаний — наш ответ.

    // Ограничиваем бюджет: не более 80% от основного timeBudget
    // (чтобы оставить запас CPU для других задач).
    double ponderBudget = timeoutFor(settings.strength) * 0.8;

    // Сначала: какие ходы может сделать соперник?
    MatchState root = toMatchState(s, k);
    MoveBuffer buf;
    int n = genLegalMoves(root, buf);
    if (n == 0) {
        ponderActive_.store(false);
        return;
    }

    // Сортируем ходы соперника по эвристике (Task 1) — топ-3 наиболее вероятны.
    CardMask oppHand = root.hands[toIdx(Player::Opp)];
    orderMoves(buf, n, root, oppHand);

    int topN = std::min(3, n);

    for (int i = 0; i < topN; ++i) {
        if (ponderStopFlag_.load()) break;

        // Применяем ход соперника к копии состояния.
        MatchState afterOpp = root;
        if (!applyMove(afterOpp, buf[i])) continue;

        // Преобразуем back в GameState для повторного вызова decide().
        // Это сложно — GameState и MatchState имеют разные структуры.
        // Поскольку мы внутри Bot, у нас нет публичного API для этого.
        //
        // РЕШЕНИЕ: считаем наш лучший ход напрямую через ISMCTS/Endgame
        // в позиции afterOpp (где уже ход соперника применён).
        //
        // ВАЖНО: afterOpp имеет viewpoint=Me (мы ходим следующими).

        // Создаём Knowledge для afterOpp.
        Knowledge k2 = k;
        k2.myHand = afterOpp.hands[toIdx(Player::Me)];
        k2.tableKnown = 0;
        for (int j = 0; j < afterOpp.tableLen; ++j) {
            k2.tableKnown |= cardBit(afterOpp.table[j].attack);
            if (afterOpp.table[j].defended)
                k2.tableKnown |= cardBit(afterOpp.table[j].defense);
        }
        k2.discard = afterOpp.discard;
        k2.oppHandCount = handSize(afterOpp.hands[toIdx(Player::Opp)]);
        k2.deckRemaining = afterOpp.deckRemaining;
        k2.trump = afterOpp.trump;
        k2.recomputeProbs();

        // Считаем ход через тот же decide() — но БЕЗ рекурсивного pondering.
        // Поэтому используем внутренний поиск напрямую.
        DecisionStats stats{};
        Move ourMove;

        if (afterOpp.deckRemaining == 0) {
            // Endgame.
            CardMask pool = k2.unknownPool();
            afterOpp.hands[toIdx(Player::Opp)] =
                sampleOppHandForEndgame(pool, k2.oppHandCount);
            afterOpp.deck = 0;
            EndgameLimits lim;
            lim.timeBudgetSec = ponderBudget / std::max(1, topN);
            EndgameResult er = bestEndgameMove(afterOpp, Player::Me, lim,
                                                &ponderStopFlag_);
            ourMove = er.move;
            stats.mode = "Ponder:Endgame";
            stats.depthReached = er.depthReached;
            stats.solved = er.solved;
            stats.timeMs = er.timeMs;
            stats.playouts = er.nodes;
        } else {
            // ISMCTS.
            IsmctsLimits lim;
            lim.timeBudgetSec = ponderBudget / std::max(1, topN);
            lim.numThreads = settings.numThreads;
            PolicyValueNet* net = hasRealNet_ ? net_.get() : nullptr;
            IsmctsResult r = runIsmcts(afterOpp, k2, lim, &ponderStopFlag_,
                                        Player::Me, net);
            ourMove = r.move;
            stats.mode = "Ponder:ISMCTS";
            stats.playouts = r.playouts;
            stats.winrate = r.winrate;
            stats.timeMs = r.timeMs;
        }

        if (ponderStopFlag_.load()) break;

        // Сохраняем в кеш: ключ = hash состояния afterOpp.
        // При checkPonderCache(s_new, k_new) мы вычислим hash для s_new
        // и сравним — если совпал, вернём ourMove.
        uint64_t h = 0;
        h ^= k2.myHand * 0x100000001B3ULL;
        h ^= k2.tableKnown * 0xC2B2AE3D27D4EB4FULL;
        h ^= (uint64_t)afterOpp.trump << 16;
        h ^= (uint64_t)afterOpp.turn << 24;
        h ^= (uint64_t)afterOpp.phase << 28;
        h ^= (uint64_t)afterOpp.attacker << 32;

        PonderEntry entry;
        entry.move = ourMove;
        entry.stats = stats;
        entry.predictedStateHash = h;

        {
            std::lock_guard<std::mutex> lk(ponderMtx_);
            // Ограничиваем кеш 32 записями — больше не нужно.
            if (ponderCache_.size() > 32) ponderCache_.clear();
            ponderCache_[h] = entry;
        }
    }

    ponderActive_.store(false);
}

Move Bot::checkPonderCache(const GameState& s, const Knowledge& k,
                            DecisionStats* statsOut) {
    uint64_t h = ponderHash(s, k);
    std::lock_guard<std::mutex> lk(ponderMtx_);
    auto it = ponderCache_.find(h);
    if (it == ponderCache_.end()) {
        // Нет попадания в кеш.
        if (statsOut) statsOut->mode = "Ponder:MISS";
        return Move{ Action::Pass, Card{}, Card{}, false, {} };
    }

    // Попадание! Возвращаем предрассчитанный ход.
    if (statsOut) *statsOut = it->second.stats;
    if (statsOut) statsOut->timeMs = 0.0;  // мгновенно
    Move m = it->second.move;

    // Очищаем кеш после использования (предсказание одноразовое).
    ponderCache_.erase(it);

    return m;
}

void Bot::stopPondering() {
    ponderStopFlag_.store(true);
    if (ponderThread_.joinable()) {
        ponderThread_.join();
    }
    ponderActive_.store(false);
    std::lock_guard<std::mutex> lk(ponderMtx_);
    ponderCache_.clear();
}

// ============================================================================
// Task 7: Endgame Database API
// ============================================================================

bool Bot::loadEndgameDB(const std::string& path) {
    bool ok = endgameDB_.load(path);
    if (ok) {
        std::fprintf(stderr, "[Bot] EndgameDB загружена: %zu записей из %s\n",
                     endgameDB_.size(), path.c_str());
    } else {
        std::fprintf(stderr, "[Bot] Не удалось загрузить EndgameDB из %s\n",
                     path.c_str());
    }
    return ok;
}

} // namespace durakk
