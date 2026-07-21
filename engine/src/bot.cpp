#include "bot.h"

#include "endgame.h"
#include "ismcts.h"
#include "rules_fast.h"
#include "rules.h"

#include <cstdio>
#include <random>

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
    m.pairsLimit = 6;
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

} // namespace durakk
