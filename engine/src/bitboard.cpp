#include "bitboard.h"

namespace durakk {

std::vector<Card> maskToCards(CardMask m) {
    std::vector<Card> out;
    out.reserve(popCount(m));
    forEachCard(m, [&](Card c) { out.push_back(c); });
    return out;
}

CardMask cardsToMask(const std::vector<Card>& cards) {
    CardMask m = 0;
    for (Card c : cards) m = maskAdd(m, c);
    return m;
}

} // namespace durakk
