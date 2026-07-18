#pragma once

#include "bitboard.h"
#include "card.h"

#include <cstdint>

namespace durakk {

// «Абсолютная память» бота: что профессиональный игрок точно знает о партии.
//
// В дураке игрок не видит руку соперника и порядок колоды, но ТОЧНО знает:
//   • свою руку;
//   • козырь;
//   • каждую ушедшую в отбой (биту) карту;
//   • какие карты соперник ЗАБРАЛ со стола (он их видел, когда соперник «брал»).
// Отсюда методом исключения строится «неизвестный пул» — множество карт, чьё
// расположение (у соперника / в колоде) боту неизвестно. ISMCTS семплирует
// из этого пула детерминизации.
struct Knowledge {
    CardMask myHand       = 0;  // моя рука (точно)
    CardMask discard      = 0;  // все карты, ушедшие в отбой (память)
    CardMask oppKnownTaken = 0; // карты, которые соперник ЗАБРАЛ (видели на столе → у него)
    CardMask tableKnown   = 0;  // карты, лежащие сейчас на столе (видим все)

    Suit trump = Suit::Spades;

    int oppHandCount  = 0;  // сколько карт сейчас у соперника
    int deckRemaining = 0;  // сколько карт осталось в колоде

    // Множество карт, чьё местоположение боту НЕ известно (ни у меня, ни в бите,
    // ни у соперника-известно-взятых, ни на столе). Эти карты распределены между
    // рукой соперника и колодой в неизвестной пропорции.
    CardMask unknownPool() const {
        // unknown = всё − (моя рука ∪ бита ∪ взято соперником ∪ стол)
        CardMask known = myHand | discard | oppKnownTaken | tableKnown;
        return (~known) & kFullDeckMask;
    }

    // Содержит ли неизвестный пул достаточно карт для раздачи сопернику.
    bool canSample() const {
        return popCount(unknownPool()) >= oppHandCount;
    }

    // Проверка целостности: ни одна карта не должна встречаться в двух местах.
    // Возвращает true, если множества не пересекаются.
    bool consistent() const {
        CardMask a = myHand & discard;
        CardMask b = myHand & oppKnownTaken;
        CardMask c = discard & oppKnownTaken;
        CardMask d = (myHand | discard | oppKnownTaken) & tableKnown;
        return (a | b | c | d) == 0;
    }

    // Очистка.
    void reset() {
        myHand = discard = oppKnownTaken = tableKnown = 0;
        oppHandCount = deckRemaining = 0;
        trump = Suit::Spades;
    }
};

} // namespace durakk
