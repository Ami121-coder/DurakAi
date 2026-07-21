#include "ttable.h"

#include <random>

namespace durakk {

namespace {

// Zobrist-ключи: по случайному 64-битному числу на (игрок, карта) и на фазу/ход.
// Ленивая инициализация — один раз на процесс.
struct Zobrist {
    uint64_t piece[2][36] = {};   // [игрок][индекс карты] — кто держит карту
    // FIX #9: позиционные ключи для стола — учитываем порядок пар.
    uint64_t tableAtkPos[6][36] = {};  // [позиция пары][индекс атаки]
    uint64_t tableDefPos[6][36] = {};  // [позиция пары][индекс защиты]
    uint64_t trump[4] = {};       // козырь
    uint64_t attacker = 0;        // чья роль атакующего
    uint64_t turn = 0;            // чей ход
    uint64_t phaseAttack = 0;
    uint64_t phaseDefense = 0;
    uint64_t phasePursue = 0;
    // FIX #9: настройки и счётчики, влияющие на легальные ходы.
    uint64_t firstTrick = 0;
    uint64_t transferEnabled = 0;
    uint64_t flashEnabled = 0;
    uint64_t pairsLimitKey = 0;   // pairsLimit × позиционное значение
    uint64_t deckRemainingKey = 0;
    uint64_t pairLimitMult[8] = {}; // для pairsLimit ∈ {0..7}
    uint64_t deckRemainingMult[40] = {}; // для deckRemaining ∈ {0..39}

    Zobrist() {
        std::mt19937_64 rng(0xD00D1ECAFE42ULL);
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < 36; ++i) piece[p][i] = rng();
        for (int pos = 0; pos < 6; ++pos) {
            for (int i = 0; i < 36; ++i) {
                tableAtkPos[pos][i] = rng();
                tableDefPos[pos][i] = rng();
            }
        }
        for (int i = 0; i < 4; ++i) trump[i] = rng();
        attacker = rng();
        turn = rng();
        phaseAttack = rng();
        phaseDefense = rng();
        phasePursue = rng();
        firstTrick = rng();
        transferEnabled = rng();
        flashEnabled = rng();
        for (int i = 0; i < 8; ++i) pairLimitMult[i] = rng();
        for (int i = 0; i < 40; ++i) deckRemainingMult[i] = rng();
    }
};

const Zobrist& zobrist() {
    static Zobrist z;
    return z;
}

} // namespace

uint64_t computeHash(const MatchState& s) {
    const Zobrist& z = zobrist();
    uint64_t h = 0;

    // Карты в руках.
    CardMask h0 = s.hands[0], h1 = s.hands[1];
    while (h0) {
        CardMask bit = h0 & (~h0 + 1); h0 ^= bit;
        int idx;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
        idx = __builtin_ctzll(bit);
#endif
        h ^= z.piece[0][idx];
    }
    while (h1) {
        CardMask bit = h1 & (~h1 + 1); h1 ^= bit;
        int idx;
#if defined(_MSC_VER)
        unsigned long ul; _BitScanForward64(&ul, bit); idx = (int)ul;
#else
        idx = __builtin_ctzll(bit);
#endif
        h ^= z.piece[1][idx];
    }

    // Стол (позиционно — FIX #9: порядок пар важен для верхнего непобитого ранга).
    for (int i = 0; i < s.tableLen; ++i) {
        h ^= z.tableAtkPos[i][cardIndex(s.table[i].attack)];
        if (s.table[i].defended)
            h ^= z.tableDefPos[i][cardIndex(s.table[i].defense)];
    }

    h ^= z.trump[static_cast<int>(s.trump)];
    h ^= (s.attacker == Player::Me) ? z.attacker : 0;
    h ^= (s.turn == Player::Opp) ? z.turn : 0;
    switch (s.phase) {
        case MatchPhase::Attack:  h ^= z.phaseAttack;  break;
        case MatchPhase::Defense: h ^= z.phaseDefense; break;
        case MatchPhase::Pursue:  h ^= z.phasePursue;  break;
        case MatchPhase::Done:    break;
        case MatchPhase::GameOver:break;
    }
    if (s.firstTrick)       h ^= z.firstTrick;
    if (s.transferEnabled)  h ^= z.transferEnabled;
    if (s.flashEnabled)     h ^= z.flashEnabled;
    // pairsLimit ∈ {5, 6} по правилам, но берём с запасом.
    int pl = static_cast<int>(s.pairsLimit);
    if (pl >= 0 && pl < 8) h ^= z.pairLimitMult[pl];
    int dr = static_cast<int>(s.deckRemaining);
    if (dr >= 0 && dr < 40) h ^= z.deckRemainingMult[dr];
    return h;
}

} // namespace durakk
