#pragma once

// Таблица перестановок (transposition table) для эндшпильного минимакса.
//
// Хеширует позицию (MatchState без колоды) в значение оценки. Используется
// для отсечения повторных поддеревьев в α-β поиске — эндшпили в дураке часто
// содержат одинаковые позиции, достигаемые разным порядком ходов.
//
// Реализация простая: unordered_map с Zobrist-хешем. Размер эндшпилей невелик,
// поэтому-memory не критичен; при необходимости легко заменить на фикс. кэш.

#include "match.h"

#include <cstdint>
#include <unordered_map>

namespace durakk {

// Zobrist-хеш позиции. Заполняется incrementally не (пока) — здесь простой полный пересчёт.
// FIX #9: учитывает firstTrick, transferEnabled, flashEnabled, pairsLimit,
// deckRemaining, а также порядок пар (через позиционный индекс) — иначе
// разные позиции давали одинаковый хеш и TT возвращал ошибочные оценки.
uint64_t computeHash(const MatchState& s);

// Запись в таблице перестановок.
struct TTEntry {
    int   depth;     // глубина, на которой искали
    int   value;     // оценка из позиции для атакующего-на-ходе (в «победах»)
    uint8_t flag;    // 0=exact, 1=lower bound, 2=upper bound
    Move  best;      // лучший ход (для упорядочения при следующем заходе)
    bool  hasBest;
};

class TTable {
public:
    void store(uint64_t key, const TTEntry& e) { table_[key] = e; }
    const TTEntry* probe(uint64_t key) const {
        auto it = table_.find(key);
        return it == table_.end() ? nullptr : &it->second;
    }
    void clear() { table_.clear(); }
    size_t size() const { return table_.size(); }

private:
    std::unordered_map<uint64_t, TTEntry> table_;
};

} // namespace durakk
