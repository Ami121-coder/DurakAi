#pragma once
#include <cstdint>

using CardMask = uint64_t;

namespace bb {

constexpr int NUM_CARDS = 36;
constexpr int NUM_RANKS = 9;
constexpr int NUM_SUITS = 4;

inline int cardIndex(int rank, int suit) { return (rank - 6) * 4 + suit; }
inline int rankOf(int idx) { return idx / 4 + 6; }
inline int suitOf(int idx) { return idx % 4; }

inline CardMask cardBit(int rank, int suit) { return 1ULL << cardIndex(rank, suit); }
inline CardMask cardBitIdx(int idx) { return 1ULL << idx; }

inline int popcount(CardMask m) {
#ifdef _MSC_VER
    return (int)__popcnt64(m);
#else
    return __builtin_popcountll(m);
#endif
}

inline CardMask suitMask(int suit) {
    CardMask m = 0;
    for (int r = 0; r < 9; ++r) m |= (1ULL << (r * 4 + suit));
    return m;
}

inline CardMask rankMask(int rank) {
    return 0xFULL << ((rank - 6) * 4);
}

inline CardMask trumpMask(int trumpSuit) { return suitMask(trumpSuit); }

inline bool beats(int atkIdx, int defIdx, int trumpSuit) {
    int aR = rankOf(atkIdx), aS = suitOf(atkIdx);
    int dR = rankOf(defIdx), dS = suitOf(defIdx);
    if (dS == trumpSuit && aS != trumpSuit) return true;
    if (dS == aS && dR > aR) return true;
    return false;
}

inline int lowestBit(CardMask m) {
    unsigned long idx;
#ifdef _MSC_VER
    _BitScanForward64(&idx, m);
#else
    idx = __builtin_ctzll(m);
#endif
    return (int)idx;
}

inline int popLowest(CardMask& m) {
    int idx = lowestBit(m);
    m &= m - 1;
    return idx;
}

// Случайный бит из маски (используется на CPU)
inline int randomBit(CardMask m, unsigned& rngState) {
    int cnt = popcount(m);
    if (cnt == 0) return -1;
    rngState = rngState * 1664525u + 1013904223u;
    int pick = (int)((rngState >> 16) % cnt);
    CardMask tmp = m;
    for (int i = 0; i < pick; ++i) tmp &= tmp - 1;
    return lowestBit(tmp);
}

} // namespace bb