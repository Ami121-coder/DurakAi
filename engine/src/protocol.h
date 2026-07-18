#pragma once

// JSON-протокол IPC (одна строка = один JSON-объект).
//
// Команды (stdin):
//   {"cmd":"setState",  ...поля состояния и памяти...}
//   {"cmd":"decide",    "strength":"normal"}   // fast|normal|deep
//   {"cmd":"validate",  "action":..., "rank":.., "suit":.., "target":{r,s}}
//   {"cmd":"legalMoves"}
//   {"cmd":"ping"}
//
// setState — расширенный формат (с «абсолютной памятью» бота):
//   {
//     "trump":"spades",
//     "deckSize":36, "deckCount":18,
//     "myHand":[{r,s}, ...],          // моя рука (точно)
//     "discard":[{r,s}, ...],         // карты, ушедшие в отбой (память бота)
//     "oppTaken":[{r,s}, ...],        // карты, которые соперник ЗАБРАЛ
//     "oppHandCount":5,
//     "table":[{"attack":{r,s},"defense":{r,s}|null}, ...],
//     "attacker":"me|opp","turn":"me|opp","phase":"attack|defense",
//     "firstTrick":true,
//     "transferEnabled":true, "flashEnabled":false, "firstTrickLimit":5
//   }
//
// Ответ decide:
//   {"ok":true, "action":"defend", "rank":"8","suit":"clubs",
//    "target":{"r":"7","s":"clubs"}, "reason":"...",
//    "stats":{"mode":"ISMCTS","playouts":28431,"winrate":0.62,"timeMs":2031,
//            "depthReached":0,"solved":false}}

#include "bot.h"
#include "game_state.h"
#include "knowledge.h"
#include "move.h"
#include "rules.h"

#include <nlohmann/json.hpp>
#include <string>

namespace durakk {

using json = nlohmann::json;

// ---------- Карты ----------
json cardToJson(Card c);
Card cardFromJson(const json& j);

// ---------- Ход ----------
json moveToJson(const Move& m);
std::string actionToString(Action a);

// ---------- Сила ----------
Strength strengthFromString(const std::string& s);

// ---------- Парсинг состояния и памяти ----------
// Заполняет state (наблюдаемая позиция) и knowledge (абсолютная память).
// Бросает std::runtime_error при ошибках.
void parseState(const json& j, GameState& state, Knowledge& knowledge);

// ---------- Кодирование статистики ----------
json statsToJson(const DecisionStats& st);

// ---------- Легальные ходы ----------
json legalMovesToJson(const GameState& s);

} // namespace durakk
