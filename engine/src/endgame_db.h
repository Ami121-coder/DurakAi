#pragma once

// ============================================================================
// endgame_db.h — Endgame Database через Retrograde Analysis (Task 7)
//
// Идея: заранее просчитать все эндшпильные позиции с суммарно ≤N карт
// (N=6) при пустой колоде и пустом столе. Для каждой позиции хранить
// правильный ход (или win/draw/loss). Во время игры — O(1) lookup.
//
// Размер БД для N=6 (только старт кона, стол пуст):
//   - Распределение карт: для каждой пары (k_мое, k_опп) с k_мое + k_опп ≤ 6
//     и k_мое, k_опп ≥ 1 (нельзя 0 — это конец игры)
//   - C(36, k_мое) × C(36-k_мое, k_опп) позиций
//   - × 4 масти козыря × 2 attacker × 2 phase (Attack/Defense) ≈
//     Суммарно ~1.3 млрд позиций — слишком много для N=6
//
// УПРОЩЕНИЕ: делаем БД для N=4 (суммарно ≤4 карт при deck=0):
//   - C(36,2)×C(34,2) × 4 × 2 × 2 ≈ 14M позиций
//   - 4 байта на запись (best move + win/draw/loss) = ~56 MB
//   - Это реально построить и хранить в RAM
//
// Формат записи (4 байта):
//   bit 0-7:   best card index (0..35, 36 = Take, 37 = Done)
//   bit 8-15:  target card index (для Defend) или 0xFF если нет
//   bit 16-23: action (Attack/Defend/Transfer/Toss/Take/Done/Pass)
//   bit 24-31: result (0=unknown, 1=win, 2=loss, 3=draw, 4=win_forced, 5=loss_forced)
//
// Алгоритм построения (retrograde):
//   1. Все терминальные позиции (один игрок без карт) → помечаем как win/loss.
//   2. Для каждой нетерминальной позиции:
//      - Если хотя бы один ход ведёт в позицию, помеченную как loss для
//        соперника → помечаем как win.
//      - Если все ходы ведут в позиции, помеченные как win для соперника →
//        помечаем как loss.
//      - Иначе — draw.
//   3. Повторяем, пока не перестанут появляться новые пометки.
//
// Построение запускается ОДИН раз (build_endgame_db.py), результат —
// бинарный файл engine/data/endgame_db.bin (~56 MB), который загружается
// в RAM при старте бота.
// ============================================================================

#include "match.h"
#include "move.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace durakk {

struct EndgameDBEntry {
    uint8_t bestCardIdx;     // 0..35 (карта), 36 (Take), 37 (Done), 0xFF (unknown)
    uint8_t targetCardIdx;   // 0..35 для Defend, иначе 0xFF
    uint8_t action;          // Action enum value
    uint8_t result;          // 0=unknown, 1=win, 2=loss, 3=draw
};

class EndgameDB {
public:
    EndgameDB() = default;

    // Загрузить БД из бинарного файла. Возвращает false при ошибке.
    bool load(const std::string& path);

    // Сохранить БД в бинарный файл.
    bool save(const std::string& path) const;

    // Проверить, загружена ли БД.
    bool isReady() const { return ready_; }

    // Lookup: возвращает best move для заданной позиции.
    // s — состояние с deck=0, table пустым, суммарно ≤4 карт на руках.
    // Если позиция не в БД — возвращает result=0 (unknown) и action=Pass.
    EndgameDBEntry lookup(const MatchState& s) const;

    // Размер БД (число записей).
    size_t size() const { return entries_.size(); }

    // Добавить/обновить запись (для построения БД).
    void put(uint64_t key, const EndgameDBEntry& e) {
        entries_[key] = e;
        ready_ = true;
    }

    // Пустая БД.
    void clear() {
        entries_.clear();
        ready_ = false;
    }

private:
    // Ключ — Zobrist-подобный хеш позиции.
    // Использует computeHash() из ttable.h + дополнительный хеш для атакующего.
    std::unordered_map<uint64_t, EndgameDBEntry> entries_;
    bool ready_ = false;
};

// Построить БД для эндшпилей с суммарно ≤maxTotal карт при deck=0.
// Рекомендуется maxTotal=4 (размер ~56 MB).
// Возвращает число записей.
size_t buildEndgameDB(EndgameDB& db, int maxTotal = 4);

} // namespace durakk
