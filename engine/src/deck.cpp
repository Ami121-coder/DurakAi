#include "deck.h"

namespace durakk {

Rank minRank(DeckSize size) {
    switch (size) {
        case DeckSize::Cards24: return Rank::Nine;
        case DeckSize::Cards36: return Rank::Six;
        case DeckSize::Cards52: return Rank::Six; // в 52-карточной тоже от шестёрок
    }
    return Rank::Six;
}

std::vector<Card> fullDeck(DeckSize size) {
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
