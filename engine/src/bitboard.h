#pragma once

// Битборды (bitboards) для 36-карточной колоды дурака.
//
// Вся колода помещается в одно 64-битное слово: каждой карте — свой бит.
// Это позволяет делать операции над множествами карт побитово
// (AND / OR / XOR / NOT) за один такт процессора, что критично для скорости
// ISMCTS (тысячи playout в секунду на Ryzen 7500F).
//
// Кодирование бита:
//   index = suitIndex(Rank,Suit) = (int(Suit) * 9) + (int(Rank) - int(Rank::Six))
//   где suitIndex ∈ [0,3], rankIndex ∈ [0,8].
//   Так, шестёрка треф — бит 0, туз пик — бит 35. Младшие 36 бит используются.

#include "card.h"

#include <cstdint>
#include <vector>

// Интринзики компилятора для popcnt / bit-scan. Доступны на MSVC, GCC, Clang.
#if defined(_MSC_VER)
#include <intrin.h> // __popcnt64, _BitScanForward64
#pragma intrinsic(__popcnt64, _BitScanForward64)
#endif

namespace durakk {

using CardMask = uint64_t;

// Маска «все 36 карт». Старшие 28 бит всегда 0.
constexpr CardMask kFullDeckMask = (uint64_t(1) << 36) - 1;

// Индекс карты в битборде [0..35].
constexpr int cardIndex(Suit s, Rank r) {
    return static_cast<int>(s) * 9 + (static_cast<int>(r) - static_cast<int>(Rank::Six));
}
constexpr int cardIndex(Card c) { return cardIndex(c.suit, c.rank); }
constexpr Card indexToCard(int idx) {
    return Card{ static_cast<Rank>(static_cast<int>(Rank::Six) + (idx % 9)),
                 static_cast<Suit>(idx / 9) };
}

// Бит одной карты.
constexpr CardMask cardBit(Suit s, Rank r) { return uint64_t(1) << cardIndex(s, r); }
constexpr CardMask cardBit(Card c)         { return uint64_t(1) << cardIndex(c); }

// ---------- Множественные операции (inline, ~1 такт) ----------

constexpr inline CardMask maskAdd(CardMask m, Card c)      { return m | cardBit(c); }
constexpr inline CardMask maskRemove(CardMask m, Card c)   { return m & ~cardBit(c); }
constexpr inline bool     maskHas(CardMask m, Card c)      { return (m & cardBit(c)) != 0; }
constexpr inline CardMask maskAnd(CardMask a, CardMask b)  { return a & b; }
constexpr inline CardMask maskOr(CardMask a, CardMask b)   { return a | b; }
constexpr inline CardMask maskXor(CardMask a, CardMask b)  { return a ^ b; }
constexpr inline CardMask maskComplement(CardMask m)       { return (~m) & kFullDeckMask; }
constexpr inline bool     maskEmpty(CardMask m)            { return m == 0; }

// Количество карт в маске. Использует popcnt (нативная инструкция).
inline int popCount(CardMask m) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(m));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(m);
#else
    // Запасной (медленный) путь.
    int n = 0;
    while (m) { m &= m - 1; ++n; }
    return n;
#endif
}

// Размер руки = popCount (алиас для читаемости).
inline int handSize(CardMask hand) { return popCount(hand); }

// ---------- Итерация по картам маски (в порядке возрастания индекса) ----------

// Вызывает f(Card) для каждой карты маски. f должно быть дешёвым — hot path.
template <typename F>
inline void forEachCard(CardMask m, F&& f) {
    while (m) {
        int idx;
#if defined(_MSC_VER)
        unsigned long ul;
        _BitScanForward64(&ul, m);
        idx = static_cast<int>(ul);
#elif defined(__GNUC__) || defined(__clang__)
        idx = __builtin_ctzll(m);
#else
        // Fallback: найти младший бит вручную.
        idx = 0;
        while (!(m & 1)) { m >>= 1; ++idx; }
#endif
        f(indexToCard(idx));
        m &= m - 1; // сбросить младший установленный бит
    }
}

// Собрать маску в вектор карт (для отладки / IPC).
std::vector<Card> maskToCards(CardMask m);

// Маска из вектора карт.
CardMask cardsToMask(const std::vector<Card>& cards);

} // namespace durakk
