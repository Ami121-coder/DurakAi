#pragma once

#include <string>
#include <cstdint>

namespace durakk {

// Масть карты. Порядок условный — козырь задаётся отдельно, масти между собой
// не упорядочены; сравнивается только достоинство карт одной масти.
enum class Suit : uint8_t {
    Clubs = 0,     // ♣ Трефы
    Diamonds = 1,  // ♦ Бубны
    Hearts = 2,    // ♥ Червы
    Spades = 3,    // ♠ Пики
};

// Достоинство карты (36-карточная колода: 6..Туз).
// Значения упорядочены по старшинству, можно сравнивать через <.
enum class Rank : uint8_t {
    Six   = 6,
    Seven = 7,
    Eight = 8,
    Nine  = 9,
    Ten   = 10,
    Jack  = 11,   // В
    Queen = 12,   // Д
    King  = 13,   // К
    Ace   = 14,   // Т
};

// Карта: масть + достоинство.
struct Card {
    Rank rank;
    Suit suit;

    bool operator==(const Card& o) const { return rank == o.rank && suit == o.suit; }
    bool operator!=(const Card& o) const { return !(*this == o); }

    // Для использования Card как ключа в std::map / set.
    bool operator<(const Card& o) const {
        if (rank != o.rank) return rank < o.rank;
        return suit < o.suit;
    }
};

// Утилиты сериализации (используются и в IPC, и в отладке).
const char* suitSymbol(Suit s);
const char* rankSymbol(Rank r);

Suit   suitFromString(const std::string& s);
Rank   rankFromString(const std::string& r);

// Карта старше другой с учётом козыря.
// Возвращает true, если `defender` бьёт `attacker`.
bool beats(Card attacker, Card defender, Suit trump);

} // namespace durakk
