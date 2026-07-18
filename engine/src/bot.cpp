#include "bot.h"

#include "endgame.h"
#include "ismcts.h"
#include "rules_fast.h"
#include "rules.h"

#include <cstdio>

namespace durakk {

namespace {

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

Bot::Bot() : net_(std::make_unique<RandomNet>()) {}

Move Bot::decide(const GameState& s, const Knowledge& k,
                 const SearchSettings& settings, DecisionStats* statsOut) {
    DecisionStats stats{};
    MatchState root = toMatchState(s, k);

    // ---------- Эндшпиль: колода пуста → minimax α-β (perfect information) ----------
    if (s.deck.remaining == 0) {
        // В эндшпиле рука соперника = все неизвестные карты. Заполним её явно,
        // т.к. минимакс требует perfect information.
        CardMask oppHand = k.unknownPool(); // всё, что не у меня и не в бите → у соперника
        root.hands[1] = oppHand;
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
    IsmctsLimits lim;
    lim.timeBudgetSec = timeoutFor(settings.strength);
    lim.numThreads = settings.numThreads;
    IsmctsResult r = runIsmcts(root, k, lim, nullptr, Player::Me);

    stats.mode = "ISMCTS";
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
