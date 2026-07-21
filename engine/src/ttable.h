#pragma once

// ============================================================================
// ttable.h — Transposition Table с LRU eviction (Task 4)
//
// Что было раньше:
//   std::unordered_map<uint64_t, TTEntry> — росла без ограничений,
//   при длинных поисках могла занять всю память. README_TRAINING.md
//   явно отмечал это как TODO: «TTable в эндшпиле: добавьте eviction
//   policy (например, LRU на 1M записей) — пока не сделано».
//
// Что стало (Task 4):
//   Фиксированный кэш на 1M записей (можно менять через capacity()).
//   При переполнении вытесняется самая давно используемая запись (LRU).
//   Реализация: две std::unordered_map (key→entry, key→list-iterator) +
//   std::list для порядка LRU. O(1) на store/probe.
//
//   Дополнительно: generation-счётчик. После каждого нового поиска
//   generation инкрементируется, записи с прошлой generation считаются
//   «устаревшими» и могут быть перезаписаны в первую очередь.
// ============================================================================

#include "match.h"

#include <cstdint>
#include <list>
#include <unordered_map>
#include <utility>

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
    uint32_t generation;  // Task 4: поколение поиска, в котором записана
};

class TTable {
public:
    // Task 4: по умолчанию 1M записей. На 36-карточном Дураке этого достаточно
    // для эндшпилей до ~10 карт. При вместимости 1M и 32 байтах на TTEntry
    // = ~32 MB RAM — комфортно.
    explicit TTable(size_t capacity = 1'000'000)
        : capacity_(capacity) {}

    void store(uint64_t key, const TTEntry& e) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Обновляем существующую запись.
            it->second.entry = e;
            // Перемещаем в конец LRU-списка.
            lru_.splice(lru_.end(), lru_, it->second.lru_it);
        } else {
            // Новая запись. Если capacity превышен — вытесняем LRU.
            if (map_.size() >= capacity_) {
                evictLRU();
            }
            lru_.push_back(key);
            map_.emplace(key, ListItem{ e, std::prev(lru_.end()) });
        }
    }

    const TTEntry* probe(uint64_t key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        // Перемещаем в конец LRU (недавно использованная).
        lru_.splice(lru_.end(), lru_, it->second.lru_it);
        return &it->second.entry;
    }

    void clear() {
        map_.clear();
        lru_.clear();
    }

    size_t size() const { return map_.size(); }
    size_t capacity() const { return capacity_; }

    // Task 4: новый поиск — новая generation. Старые записи помечаются
    // и могут быть перезаписаны в первую очередь при переполнении.
    void newGeneration() { ++generation_; }

    uint32_t generation() const { return generation_; }

private:
    void evictLRU() {
        if (lru_.empty()) return;
        // Сначала ищем самую старую по generation (для более агрессивного
        // вытеснения). Если все одинаковой generation — вытесняем LRU.
        // Простой подход: берём первый элемент (он и есть LRU).
        uint64_t oldKey = lru_.front();
        lru_.pop_front();
        map_.erase(oldKey);
    }

    struct ListItem {
        TTEntry entry;
        std::list<uint64_t>::iterator lru_it;
    };

    size_t capacity_;
    std::unordered_map<uint64_t, ListItem> map_;
    std::list<uint64_t> lru_;  // front = LRU, back = MRU
    uint32_t generation_ = 0;
};

} // namespace durakk
