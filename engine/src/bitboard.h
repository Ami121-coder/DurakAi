#pragma once
#include <cstdint>
#include <cstring>

// 36 карт: ранги 6..14 (6,7,8,9,10,J,Q,K,A), масти 0..3 (clubs,diamonds,hearts,spades)
// Индекс карты: rank_offset * 4 + suit, где rank_offset = rank - 6 (0..8)
// Итого биты 0..35

using CardMask = uint64_t; // 36 бит используются

namespace bb {

constexpr int NUM_CARDS = 36;
constexpr int NUM_RANKS = 9;  // 6..14
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

// Все карты масти
inline CardMask suitMask(int suit) {
    CardMask m = 0;
    for (int r = 0; r < 9; ++r) m |= (1ULL << (r * 4 + suit));
    return m;
}

// Все карты ранга
inline CardMask rankMask(int rank) {
    int off = (rank - 6) * 4;
    return 0xFULL << off; // 4 бита подряд
}

// Козырные карты
inline CardMask trumpMask(int trumpSuit) { return suitMask(trumpSuit); }

// Старшинство: козырь бьёт не-козырь; внутри масти — по рангу
inline bool beats(int atkIdx, int defIdx, int trumpSuit) {
    int aRank = rankOf(atkIdx), aSuit = suitOf(atkIdx);
    int dRank = rankOf(defIdx), dSuit = suitOf(defIdx);
    bool aT = (aSuit == trumpSuit);
    bool dT = (dSuit == trumpSuit);
    if (dT && !aT) return true;
    if (dSuit == aSuit && dRank > aRank) return true;
    return false;
}

// Наименьший бит
inline int lowestBit(CardMask m) {
    unsigned long idx;
#ifdef _MSC_VER
    _BitScanForward64(&idx, m);
#else
    idx = __builtin_ctzll(m);
#endif
    return (int)idx;
}

// Убрать наименьший бит, вернуть индекс
inline int popLowest(CardMask& m) {
    int idx = lowestBit(m);
    m &= m - 1;
    return idx;
}

} // namespace bb