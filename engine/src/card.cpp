#include "card.h"

#include <stdexcept>

namespace durakk {

const char* suitSymbol(Suit s) {
    switch (s) {
        case Suit::Clubs:    return "\xE2\x99\xA3"; // ♣
        case Suit::Diamonds: return "\xE2\x99\xA6"; // ♦
        case Suit::Hearts:   return "\xE2\x99\xA5"; // ♥
        case Suit::Spades:   return "\xE2\x99\xA0"; // ♠
    }
    return "?";
}

const char* rankSymbol(Rank r) {
    switch (r) {
        case Rank::Six:   return "6";
        case Rank::Seven: return "7";
        case Rank::Eight: return "8";
        case Rank::Nine:  return "9";
        case Rank::Ten:   return "10";
        case Rank::Jack:  return "J";
        case Rank::Queen: return "Q";
        case Rank::King:  return "K";
        case Rank::Ace:   return "A";
    }
    return "?";
}

Suit suitFromString(const std::string& s) {
    // Принимаем как код из IPC (clubs/diamonds/hearts/spades), так и символ.
    if (s == "clubs" || s == "\xE2\x99\xA3" || s == "c") return Suit::Clubs;
    if (s == "diamonds" || s == "\xE2\x99\xA6" || s == "d") return Suit::Diamonds;
    if (s == "hearts" || s == "\xE2\x99\xA5" || s == "h") return Suit::Hearts;
    if (s == "spades" || s == "\xE2\x99\xA0" || s == "s") return Suit::Spades;
    throw std::runtime_error("unknown suit: " + s);
}

Rank rankFromString(const std::string& r) {
    if (r == "6" || r == "six")   return Rank::Six;
    if (r == "7" || r == "seven") return Rank::Seven;
    if (r == "8" || r == "eight") return Rank::Eight;
    if (r == "9" || r == "nine")  return Rank::Nine;
    if (r == "10" || r == "ten")  return Rank::Ten;
    if (r == "J" || r == "j" || r == "jack")   return Rank::Jack;
    if (r == "Q" || r == "q" || r == "queen")  return Rank::Queen;
    if (r == "K" || r == "k" || r == "king")   return Rank::King;
    if (r == "A" || r == "a" || r == "ace")    return Rank::Ace;
    throw std::runtime_error("unknown rank: " + r);
}

bool beats(Card attacker, Card defender, Suit trump) {
    const bool aTrump = (attacker.suit == trump);
    const bool dTrump = (defender.suit == trump);

    // Козырь бьёт некозырь.
    if (dTrump && !aTrump) return true;
    // Некозырь не бьёт козырь.
    if (!dTrump && aTrump) return false;
    // Обе козырные или обе некозырные одной масти — нужна та же масть и старше.
    if (defender.suit != attacker.suit) return false;
    return defender.rank > attacker.rank;
}

} // namespace durakk
