#include "rules_fast.h"
#include "rules.h" // canBeat

#include <algorithm>
#include <cmath>
#include <climits>
#include <random>

namespace durakk {

namespace {

// Добавить карточный ход (без target) в буфер, инкрементируя счётчик.
inline void pushMove(MoveBuffer& buf, int& n, Action a, Card c) {
    if (n < kMaxMoves) {
        buf[n++] = Move{ a, c, Card{}, false, {} };
    }
}

// Маска карт руки, чей ранг присутствует на столе.
CardMask rankOnTableMask(const MatchState& s, CardMask hand) {
    // Соберём ранги, что есть на столе.
    int ranksBits = 0; // бит r (0..8) = ранг присутствует
    for (int i = 0; i < s.tableLen; ++i) {
        ranksBits |= (1 << (static_cast<int>(s.table[i].attack.rank) - 6));
        if (s.table[i].defended)
            ranksBits |= (1 << (static_cast<int>(s.table[i].defense.rank) - 6));
    }
    CardMask out = 0;
    CardMask h = hand;
    while (h) {
        CardMask bit = h & (~h + 1);
        h ^= bit;
        int idx;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
        idx = __builtin_ctzll(bit);
#endif
        Card c = indexToCard(idx);
        int r = static_cast<int>(c.rank) - 6;
        if (ranksBits & (1 << r)) out |= bit;
    }
    return out;
}

} // namespace

int genAttackMoves(const MatchState& s, MoveBuffer& buf, bool* doneOut) {
    int n = 0;
    if (doneOut) *doneOut = false;

    // Подкидывание работает и в фазе Attack, и в фазе Pursue (вдогонку).
    if (s.phase != MatchPhase::Attack && s.phase != MatchPhase::Pursue) return 0;
    if (s.turn != s.attacker) return 0;

    const CardMask hand = s.hands[toIdx(s.attacker)];
    int headroom = s.pairsHeadroom();

    if (s.tableLen == 0) {
        // Первая карта кона — любая из руки (фаза Attack только).
        CardMask h = hand;
        while (h) {
            CardMask bit = h & (~h + 1);
            h ^= bit;
            int idx;
#if defined(_MSC_VER)
            unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
            idx = __builtin_ctzll(bit);
#endif
            pushMove(buf, n, Action::Attack, indexToCard(idx));
        }
        return n;
    }

    // Подкидывание: ранги уже на столе; не больше headroom пар.
    if (headroom <= 0) {
        if (doneOut) *doneOut = (s.undefendedCount() == 0);
        // В Pursue всегда можно завершить (Pass) — flag вернёт true ниже.
        if (doneOut && s.phase == MatchPhase::Pursue) *doneOut = true;
        return 0;
    }
    CardMask candidates = rankOnTableMask(s, hand);
    CardMask h = candidates;
    while (h) {
        CardMask bit = h & (~h + 1);
        h ^= bit;
        int idx;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
        idx = __builtin_ctzll(bit);
#endif
        pushMove(buf, n, Action::Toss, indexToCard(idx));
    }
    // Готово «бито» (или завершение взятия), если все побито ИЛИ мы в Pursue.
    if (doneOut) {
        if (s.phase == MatchPhase::Pursue) {
            *doneOut = true;  // всегда можно завершить взятие через Pass
        } else {
            *doneOut = (s.undefendedCount() == 0);
        }
    }
    return n;
}

int genDefenseMoves(const MatchState& s, MoveBuffer& buf) {
    int n = 0;
    if (s.phase != MatchPhase::Defense || s.turn != s.defender()) return 0;

    const CardMask hand = s.hands[toIdx(s.defender())];

    // 1) Защита: для каждой непобитой атаки — все карты руки, которые её бьют.
    for (int i = 0; i < s.tableLen; ++i) {
        if (s.table[i].defended) continue;
        Card atk = s.table[i].attack;
        CardMask h = hand;
        while (h) {
            CardMask bit = h & (~h + 1);
            h ^= bit;
            int idx;
#if defined(_MSC_VER)
            unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
            idx = __builtin_ctzll(bit);
#endif
            Card def = indexToCard(idx);
            if (canBeat(atk, def, s.trump)) {
                if (n < kMaxMoves) {
                    buf[n++] = Move{ Action::Defend, def, atk, true, {} };
                }
            }
        }
    }

    // 2) Перевод: карты того же ранга, что верхняя непобитая атака (если разрешено
    //    и позволяет лимит). Проверку лимита делает applyMove, но мы отсечём явно,
    //    чтобы не предлагать ходы, которые точно не пройдут.
    if (s.transferEnabled) {
        Rank topRank;
        if (s.topUndefendedAttackRank(topRank)) {
            CardMask h = hand;
            while (h) {
                CardMask bit = h & (~h + 1);
                h ^= bit;
                int idx;
#if defined(_MSC_VER)
                unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
                idx = __builtin_ctzll(bit);
#endif
                Card c = indexToCard(idx);
                if (c.rank == topRank) {
                    // Проверим лимит по руке нового защищающегося (бывшего атакующего).
                    int newDefHand = popCount(s.hands[toIdx(s.attacker)]);
                    int cardsAfter = s.undefendedCount() + 1;
                    // FIX #8: лимит пар по руке НОВОГО защитника.
                    int byRules = s.firstTrick ? std::min(s.pairsLimit, 5) : s.pairsLimit;
                    int maxPairs = std::min(byRules, newDefHand);
                    if (newDefHand >= cardsAfter && s.tableLen < maxPairs) {
                        pushMove(buf, n, Action::Transfer, c);
                    }
                }
            }
        }
    }

    // 3) Взять — всегда легально в защите, если стол не пуст.
    if (s.tableLen > 0 && n < kMaxMoves) {
        buf[n++] = Move{ Action::Take, Card{}, Card{}, false, {} };
    }
    return n;
}

int genLegalMoves(const MatchState& s, MoveBuffer& buf) {
    if (s.phase == MatchPhase::Attack || s.phase == MatchPhase::Pursue) {
        bool doneOk = false;
        int n = genAttackMoves(s, buf, &doneOk);
        if (doneOk && n < kMaxMoves) {
            // В Pursue используем Pass как индикатор завершения взятия.
            // В Attack — Done как «бито».
            buf[n++] = Move{ (s.phase == MatchPhase::Pursue)
                             ? Action::Pass : Action::Done,
                             Card{}, Card{}, false, {} };
        }
        return n;
    }
    if (s.phase == MatchPhase::Defense) {
        return genDefenseMoves(s, buf);
    }
    return 0;
}

// ============================ Default policy (Smart Playout) ============================
//
// ЭТАП 1: усиленный rollout для ISMCTS.
//
// Старая версия (70% «дешевейший ход» / 30% random) не учитывала:
//   • парные ранги в руке (выгодно атаковать/переводить тем, что есть в дубле);
//   • экономию козырей (козырь в playout — слишком «дорогой» обмен);
//   • ситуацию Take vs Defend (много атак + дорогая защита → лучше Take);
//   • близость эндшпиля (deckRemaining мало → козыри ценнее, их НЕ сливать).
//
// Новый подход: для каждого хода считаем «вес предпочтительности» moveScore(),
// затем softmax-сэмплинг с температурой. Это даёт и направленность (лучшие ходы
// выбираются чаще), и разнообразие (не детерминизм, нужное для MCTS-rollout).

namespace {

inline std::mt19937_64& rng() {
    static thread_local std::mt19937_64 r{ std::random_device{}() };
    return r;
}

// Базовая «стоимость» карты (меньше = предпочтительнее сыграть в playout).
// Козырь дороже некозырного, защита козырем — ещё дороже.
int cardCost(Card c, Suit trump, Action a) {
    int base = static_cast<int>(c.rank);
    if (c.suit == trump) base += 100;
    if (a == Action::Defend && c.suit == trump) base += 50;
    return base;
}

// Сколько карт данного ранга в руке (для парных атак/переводов).
int countRankInHand(CardMask hand, Rank r) {
    int n = 0;
    CardMask h = hand;
    while (h) {
        CardMask bit = h & (~h + 1);
        h ^= bit;
        int idx;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
        idx = __builtin_ctzll(bit);
#endif
        if (indexToCard(idx).rank == r) ++n;
    }
    return n;
}

// Считает число козырей в руке.
int countTrumpsInHand(CardMask hand, Suit trump) {
    int n = 0;
    CardMask h = hand;
    while (h) {
        CardMask bit = h & (~h + 1);
        h ^= bit;
        int idx;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
        idx = __builtin_ctzll(bit);
#endif
        if (indexToCard(idx).suit == trump) ++n;
    }
    return n;
}

// «Оценка предпочтительности» хода в позиции s. Выше = лучше сыграть.
// Диапазон условный ~[0, 100]; будет преобразован в softmax-вес.
float moveScore(const MatchState& s, const Move& m) {
    const Player cur = s.turn;
    const CardMask hand = s.hands[toIdx(cur)];
    const int undefAtk = s.undefendedCount();
    const bool nearEndgame = s.deckRemaining <= 6;
    const int myTrumps = countTrumpsInHand(hand, s.trump);

    switch (m.action) {

    // ---------- ЗАЩИТА (Defend) ----------
    case Action::Defend: {
        float score = 50.0f;
        // Младшая некозырная защита — предпочтительнее.
        int cost = cardCost(m.card, s.trump, m.action);
        score -= (cost - 6) * 2.0f;            // 6 = младший ранг
        // Защита козырем — штраф, особенно в эндшпиле или когда козырей мало.
        if (m.card.suit == s.trump) {
            score -= nearEndgame ? 25.0f : 15.0f;
            if (myTrumps <= 2) score -= 15.0f; // последний козырь — не сливай
        }
        return score;
    }

    // ---------- ПЕРЕВООД (Transfer) ----------
    case Action::Transfer: {
        float score = 45.0f;
        // Перевод выгоден, если у соперника (нового защитника) мало карт:
        // cardsAfter = undefendedCount + 1; если newDefHand < cardsAfter — перевод
        // вообще невозможен. Если newDefHand лишь чуть больше — перевод давит.
        int newDefHand = popCount(s.hands[toIdx(s.attacker)]);
        int cardsAfter = undefAtk + 1;
        int slack = newDefHand - cardsAfter;
        if (slack >= 0) score += 8.0f * (3 - std::min(3, slack));
        // Перевод парной картой — особенно силён (есть дубликат для следующего
        // перевода или защиты).
        if (countRankInHand(hand, m.card.rank) >= 2) score += 12.0f;
        // Перевод козырем — рискованно (теряем козырь).
        if (m.card.suit == s.trump) {
            score -= nearEndgame ? 30.0f : 18.0f;
            if (myTrumps <= 2) score -= 15.0f;
        }
        return score;
    }

    // ---------- ВЗЯТЬ (Take) ----------
    case Action::Take: {
        // Take тем выгоднее, чем больше непобитых атак (иначе они уйдут в отбой
        // бесплатно для атакующего) и чем дороже защита.
        float score = 20.0f + undefAtk * 8.0f;
        // Но если есть хоть какая-то дешёвая защита — Take штрафуется.
        // (Грубая эвристика: если непобитых атак мало, лучше защищаться.)
        if (undefAtk <= 1) score -= 25.0f;
        // Близко к концу колоды — брать карты опасно (останутся в руке в эндшпиле).
        if (nearEndgame) score -= 10.0f;
        return score;
    }

    // ---------- АТАКА / ПОДКИДЫВАНИЕ (Attack/Toss) ----------
    case Action::Attack:
    case Action::Toss: {
        float score = 50.0f;
        // Парные ранги в руке — отлично (можно перевести/подкинуть ещё раз).
        if (countRankInHand(hand, m.card.rank) >= 2) score += 12.0f;
        // Младшие карты — предпочтительнее (не жалко потерять).
        int cost = cardCost(m.card, s.trump, m.action);
        score -= (cost - 6) * 1.5f;
        // Атака/подкидывание козырем — почти всегда плохо (теряем козырь в отбой).
        if (m.card.suit == s.trump) {
            score -= nearEndgame ? 35.0f : 22.0f;
            if (myTrumps <= 2) score -= 20.0f;
        }
        // Подкидывание старших рангов, которые соперник вряд ли побьёт — хорошо,
        // если у соперника мало карт (он будет брать). Ориентируемся по размеру
        // руки соперника в детерминизации.
        Player opp = other(cur);
        int oppHand = popCount(s.hands[toIdx(opp)]);
        if (static_cast<int>(m.card.rank) >= 12 && oppHand <= 3 && m.card.suit != s.trump) {
            score += 8.0f;
        }
        return score;
    }

    // ---------- БИТО / ПАС (Done/Pass) ----------
    case Action::Done:
    case Action::Pass:
        // Завершаем кон, если все побито — почти всегда хорошо (не даём подкинуть).
        return (undefAtk == 0) ? 70.0f : 5.0f;
    }
    return 30.0f;
}

} // namespace

bool defaultPolicy(const MatchState& s, const Move* buf, int n, Move& out) {
    if (n == 0) return false;

    // Считаем веса через softmax от moveScore.
    // Температура T: меньше = жёстче следование лучшему ходу.
    // В rollout нужна умеренная стохастичность → T=1.0 (стандартный softmax).
    constexpr float T = 1.0f;

    float scores[kMaxMoves];
    float mx = -1e18f;
    for (int i = 0; i < n; ++i) {
        scores[i] = moveScore(s, buf[i]);
        if (scores[i] > mx) mx = scores[i];
    }

    // softmax: вес_i = exp((s_i - mx) / T). Нормируем для сэмплинга.
    float weights[kMaxMoves];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        // Ограничиваем показатель, чтобы не получить overflow/inf.
        float e = (scores[i] - mx) / T;
        if (e < -30.0f) e = -30.0f;
        weights[i] = std::exp(e);
        sum += weights[i];
    }
    if (sum <= 0.0f) {
        // Фолбэк: равномерный выбор.
        std::uniform_int_distribution<int> pick(0, n - 1);
        out = buf[pick(rng())];
        return true;
    }

    // Сэмплирование по весам (inverse-CDF).
    std::uniform_real_distribution<float> u01(0.0f, sum);
    float r = u01(rng());
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += weights[i];
        if (r <= acc) {
            out = buf[i];
            return true;
        }
    }
    out = buf[n - 1]; // числовая страховка
    return true;
}

} // namespace durakk
