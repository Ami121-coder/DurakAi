#include "rules_fast.h"
#include "rules.h" // canBeat

#include <random>
#include <climits>

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

    if (s.phase != MatchPhase::Attack || s.turn != s.attacker) return 0;

    const CardMask hand = s.hands[toIdx(s.attacker)];
    int headroom = s.pairsHeadroom();

    if (s.tableLen == 0) {
        // Первая карта кона — любая из руки.
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
    // Замечание: мы генерируем кандидатов, оставляя контроль лимита applyMove'у;
    // но чтобы не плодить лишние ходы, учтём headroom по числу пар.
    if (headroom <= 0) {
        if (doneOut) *doneOut = (s.undefendedCount() == 0);
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
    // Готово «бито», если все побито.
    if (doneOut) *doneOut = (s.undefendedCount() == 0);
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
                    if (newDefHand >= cardsAfter && s.pairsHeadroom() > 0) {
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
    if (s.phase == MatchPhase::Attack) {
        bool doneOk = false;
        int n = genAttackMoves(s, buf, &doneOk);
        if (doneOk && n < kMaxMoves) {
            buf[n++] = Move{ Action::Done, Card{}, Card{}, false, {} };
        }
        return n;
    }
    if (s.phase == MatchPhase::Defense) {
        return genDefenseMoves(s, buf);
    }
    return 0;
}

// ============================ Default policy ============================

namespace {

inline std::mt19937_64& rng() {
    static thread_local std::mt19937_64 r{ std::random_device{}() };
    return r;
}

int cardWeight(Card c, Suit trump, Action a) {
    // Меньший вес = предпочтительнее в playout.
    int base = static_cast<int>(c.rank);
    int v = base + (c.suit == trump ? 100 : 0);
    // Защита козырем — особенно «дорогая», поднимем ещё.
    if (a == Action::Defend && c.suit == trump) v += 50;
    return v;
}

} // namespace

bool defaultPolicy(const MatchState& s, const Move* buf, int n, Move& out) {
    if (n == 0) return false;

    // Лёгкая взвешенная случайность: 70% — выбрать ход с минимальным «весом»
    // (младшая некозырная), 30% — равномерно случайный. Это баланс между
    // осмысленностью playout и разнообразием.
    auto& r = rng();
    std::uniform_int_distribution<int> coin(0, 9);

    if (coin(r) < 7) {
        // Выбрать «дешевлейший» ход. Take/Done/Pass считаем нейтральными.
        int bestIdx = 0;
        int bestW = INT32_MAX;
        for (int i = 0; i < n; ++i) {
            const Move& m = buf[i];
            if (m.action == Action::Take || m.action == Action::Done ||
                m.action == Action::Pass) {
                // Нейтральный вес; в защите слегка неохотно брать.
                int w = (m.action == Action::Take) ? 30 : 5;
                if (w < bestW) { bestW = w; bestIdx = i; }
                continue;
            }
            int w = cardWeight(m.card, s.trump, m.action);
            if (w < bestW) { bestW = w; bestIdx = i; }
        }
        out = buf[bestIdx];
    } else {
        std::uniform_int_distribution<int> pick(0, n - 1);
        out = buf[pick(r)];
    }
    return true;
}

} // namespace durakk
