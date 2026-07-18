#include "ttable.h"

#include <random>

namespace durakk {

namespace {

// Zobrist-ключи: по случайному 64-битному числу на (игрок, карта) и на фазу/ход.
// Ленивая инициализация — один раз на процесс.
struct Zobrist {
    uint64_t piece[2][36] = {};   // [игрок][индекс карты] — кто держит карту
    uint64_t tableAtk[36] = {};   // атакующая карта на столе
    uint64_t tableDef[36] = {};   // защитная карта на столе
    uint64_t trump[4] = {};       // козырь
    uint64_t attacker = 0;        // чья роль атакующего
    uint64_t turn = 0;            // чей ход
    uint64_t phaseAttack = 0;
    uint64_t phaseDefense = 0;

    Zobrist() {
        std::mt19937_64 rng(0xD00D1ECAFE42ULL);
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < 36; ++i) piece[p][i] = rng();
        for (int i = 0; i < 36; ++i) { tableAtk[i] = rng(); tableDef[i] = rng(); }
        for (int i = 0; i < 4; ++i) trump[i] = rng();
        attacker = rng();
        turn = rng();
        phaseAttack = rng();
        phaseDefense = rng();
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

    // Стол.
    for (int i = 0; i < s.tableLen; ++i) {
        h ^= z.tableAtk[cardIndex(s.table[i].attack)];
        if (s.table[i].defended) h ^= z.tableDef[cardIndex(s.table[i].defense)];
    }

    h ^= z.trump[static_cast<int>(s.trump)];
    h ^= (s.attacker == Player::Me) ? z.attacker : 0;
    h ^= (s.turn == Player::Opp) ? z.turn : 0;
    h ^= (s.phase == MatchPhase::Attack) ? z.phaseAttack : z.phaseDefense;
    return h;
}

} // namespace durakk
