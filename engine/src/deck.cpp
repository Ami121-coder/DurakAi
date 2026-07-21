#include "deck.h"

#include <stdexcept>

namespace durakk {

Rank minRank(DeckSize size) {
    switch (size) {
        case DeckSize::Cards24: return Rank::Nine;
        case DeckSize::Cards36: return Rank::Six;
        case DeckSize::Cards52:
            // FIX #12: 52-карточная колода официально не реализована.
            // Rank enum содержит только Six..Ace (9 значений), а bitboard.h
            // kFullDeckMask = (1<<36)-1 не вместит ранги 2..5. Возвращаем
            // Six, но в fullDeck() проверим и бросим исключение.
            return Rank::Six;
    }
    return Rank::Six;
}

std::vector<Card> fullDeck(DeckSize size) {
    if (size == DeckSize::Cards52) {
        // FIX #12: явно выбрасываем исключение вместо молчаливой генерации
        // 36-карточной колоды. Раньше это приводило к рассинхрону: UI говорил
        // «52 карты», движок работал с 36.
        throw std::runtime_error(
            "52-карточная колода не поддерживается: Rank enum содержит только "
            "Six..Ace, а bitboard.h kFullDeckMask рассчитан на 36 карт. "
            "Используйте 36- или 24-карточную колоду.");
    }
    std::vector<Card> out;
    const Rank lo = minRank(size);
    const int count = static_cast<int>(size);
    out.reserve(count);
    for (int s = 0; s < 4; ++s) {
        for (int r = static_cast<int>(lo); r <= static_cast<int>(Rank::Ace); ++r) {
            out.push_back(Card{ static_cast<Rank>(r), static_cast<Suit>(s) });
        }
    }
    return out;
}

} // namespace durakk
