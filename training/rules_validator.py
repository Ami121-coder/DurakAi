"""
rules_validator.py — строгий валидатор правил переводного дурака.

Каждая функция возвращает ValidationResult:
  { ok: bool, reason: str, rule: str, severity: str }

severity ∈ {"info", "warning", "critical"}
  • critical  — прямое нарушение правил, немедленный штраф
  • warning   — спорный ход (по правилам допустим, но подозрителен)
  • info      — просто комментарий

Правила (по ТЗ заказчика):
  §1 Подготовка: 36 карт (по умолчанию), 6 на руки, козырь — открытая карта.
  §2 Цель: первым избавиться от всех карт после окончания колоды.
  §3 Первый ход: у кого козырь наименьший; далее «под дурака».
  §4 Атака: 1+ карт одного ранга. Защита: той же мастью старше, козырь бьёт
     некозырь, козырь — только старшим козырем.
  §5 Перевод: карту того же ранга любой масти. Нельзя, если у соперника меньше
     карт, чем получится после перевода. Проездной — 1 раз за кон.
  §6 Подкидывание: только ранги со стола. Лимит 6 пар (5 в первом коне).
     Нельзя подкинуть больше, чем карт у защитника.
  §7 Завершение: «Бито» (все побито, атакующий не подкидывает) → сброс, ход к
     защитнику. «Беру»: сначала «вдогонку» (те же ранги), затем защитник забирает.
  §8 Добор: до 6 карт, первым — экс-атакующий, затем — экс-защитник.
  §9 Конец: после колоды играем остатком рук. Первый без карт — победитель.

Валидатор НЕ знает про MatchState из C++. Он работает с plain-dict снапшотом,
который строит StrictArena на каждом ходу. Это даёт независимость от
внутреннего представления движка и читаемость логов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple


# ===========================================================================
# Карты и базовые типы
# ===========================================================================

RANK_ORDER = {"6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
RANK_TO_NAME = {"6": "шестёрка", "7": "семёрка", "8": "восьмёрка", "9": "девятка",
                "10": "десятка", "J": "валет", "Q": "дама", "K": "король", "A": "туз"}
SUIT_TO_NAME = {"clubs": "трефы", "diamonds": "бубны", "hearts": "червы", "spades": "пики"}


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    rule: str = ""        # номер правила из ТЗ (например, "§4", "§6")
    severity: Severity = Severity.INFO
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "rule": self.rule,
            "severity": self.severity.value,
            "extra": self.extra,
        }


def card_str(card: dict) -> str:
    """{'r':'7','s':'clubs'} → '7♣'."""
    if not card:
        return "?"
    sym = {"clubs": "♣", "diamonds": "♦", "hearts": "♥", "spades": "♠"}.get(card["s"], "?")
    return f"{card['r']}{sym}"


def card_rank(card: dict) -> int:
    return RANK_ORDER.get(card["r"], 0)


def card_suit(card: dict) -> str:
    return card["s"]


def same_card(a: dict, b: dict) -> bool:
    return a["r"] == b["r"] and a["s"] == b["s"]


def card_in_list(c: dict, lst: List[dict]) -> bool:
    return any(same_card(c, x) for x in lst)


def beats(attacker: dict, defender: dict, trump: str) -> bool:
    """Возвращает True, если defender бьёт attacker (с учётом козыря)."""
    a_trump = (attacker["s"] == trump)
    d_trump = (defender["s"] == trump)
    if d_trump and not a_trump:
        return True
    if not d_trump and a_trump:
        return False
    if defender["s"] != attacker["s"]:
        return False
    return card_rank(defender) > card_rank(attacker)


# ===========================================================================
# Снимок состояния игры (передаётся из StrictArena)
# ===========================================================================

@dataclass
class GameStateSnapshot:
    """Снимок игры на момент хода. Plain-data для независимости от движка."""
    # Настройки
    deck_size: int = 36
    trump: str = "spades"
    transfer_enabled: bool = True
    flash_enabled: bool = False
    first_trick_limit: int = 5
    # Поле
    my_hand: List[dict] = field(default_factory=list)
    opp_hand_count: int = 0
    table: List[dict] = field(default_factory=list)   # [{attack, defense|None, defended}]
    deck_remaining: int = 0
    discard: List[dict] = field(default_factory=list)
    # Кто что делает
    attacker: str = "me"        # "me" | "opp"
    turn: str = "me"            # чей сейчас ход
    phase: str = "attack"       # "attack" | "defense" | "pursue"
    first_trick: bool = True
    flash_used_this_trick: bool = False
    # История текущего кона (для отслеживания переводов/проездного)
    transfers_this_trick: int = 0
    # FIX strict_arena: флаг прямого завершения игры (из env, если snap
    # построен через _snapshot_from_env). Если True — арена должна завершить
    # цикл ходов без проверки условий.
    is_game_over: bool = False
    winner: int = -1            # -1 = не закончена, 0 = Me, 1 = Opp


# ===========================================================================
# Проверки ходов по правилам
# ===========================================================================

def validate_attack(snap: GameStateSnapshot, card: dict) -> ValidationResult:
    """§4, §6 — атака/подкидывание."""
    if snap.phase not in ("attack", "pursue"):
        return ValidationResult(
            False, f"атака невозможна в фазе '{snap.phase}'", "§4",
            Severity.CRITICAL)

    if snap.turn != snap.attacker:
        return ValidationResult(
            False, "атакует не тот игрок (ход не атакующего)", "§4",
            Severity.CRITICAL)

    if not card_in_list(card, snap.my_hand):
        return ValidationResult(
            False, f"карты {card_str(card)} нет в руке атакующего", "§4",
            Severity.CRITICAL, {"card": card})

    # Первая карта кона — любая.
    if not snap.table:
        return ValidationResult(True, "первая карта кона", "§4", Severity.INFO)

    # Подкидывание: ранг должен быть уже на столе (§6).
    ranks_on_table = set()
    for p in snap.table:
        ranks_on_table.add(p["attack"]["r"])
        if p.get("defense"):
            ranks_on_table.add(p["defense"]["r"])
    if card["r"] not in ranks_on_table:
        return ValidationResult(
            False,
            f"подкидывать {card_str(card)} нельзя: ранг {card['r']} "
            f"отсутствует на столе (§6 — только ранги со стола)",
            "§6", Severity.CRITICAL, {"card": card})

    # Лимит пар (§6): 6 или 5 в первом коне, но не больше карт у защитника.
    by_rules = snap.first_trick_limit if snap.first_trick else 6
    defender_hand = len(snap.my_hand) if snap.attacker == "opp" else snap.opp_hand_count
    max_pairs = min(by_rules, defender_hand)
    if len(snap.table) >= max_pairs:
        return ValidationResult(
            False,
            f"превышен лимит пар ({len(snap.table)} ≥ {max_pairs}). "
            f"{'Первый кон: 5 пар' if snap.first_trick else 'Лимит 6 пар'}. "
            f"Карт у защитника: {defender_hand} (§6).",
            "§6", Severity.CRITICAL,
            {"table_len": len(snap.table), "max_pairs": max_pairs})

    return ValidationResult(True, "", "§6", Severity.INFO)


def validate_defend(snap: GameStateSnapshot, card: dict,
                    target: Optional[dict] = None) -> ValidationResult:
    """§4 — защита."""
    if snap.phase != "defense":
        return ValidationResult(
            False, f"защита невозможна в фазе '{snap.phase}'", "§4",
            Severity.CRITICAL)

    if snap.turn == snap.attacker:
        return ValidationResult(
            False, "защищается атакующий (ход должен быть у защитника)", "§4",
            Severity.CRITICAL)

    if not card_in_list(card, snap.my_hand):
        return ValidationResult(
            False, f"карты {card_str(card)} нет в руке защитника", "§4",
            Severity.CRITICAL, {"card": card})

    # Должна быть непобитая атака.
    undefended = [p for p in snap.table if not p.get("defended")]
    if not undefended:
        return ValidationResult(
            False, "неоткуда защищаться: все атаки уже побиты", "§4",
            Severity.CRITICAL)

    # Если target указан — проверим конкретную; иначе любая непобитая.
    candidates = undefended
    if target is not None:
        candidates = [p for p in undefended if same_card(p["attack"], target)]
        if not candidates:
            return ValidationResult(
                False, f"цель {card_str(target)} не найдена среди непобитых атак",
                "§4", Severity.CRITICAL, {"target": target})

    for p in candidates:
        if beats(p["attack"], card, snap.trump):
            return ValidationResult(True, "", "§4", Severity.INFO)

    return ValidationResult(
        False,
        f"картой {card_str(card)} нельзя побить ни одну непобитую атаку "
        f"(козырь: {snap.trump}, §4)",
        "§4", Severity.CRITICAL,
        {"card": card, "candidates": [p["attack"] for p in candidates]})


def validate_transfer(snap: GameStateSnapshot, card: dict) -> ValidationResult:
    """§5 — перевод."""
    if not snap.transfer_enabled:
        return ValidationResult(
            False, "перевод запрещён настройками стола", "§5",
            Severity.CRITICAL)

    if snap.phase != "defense":
        return ValidationResult(
            False, f"перевод возможен только в фазе защиты (сейчас '{snap.phase}')",
            "§5", Severity.CRITICAL)

    if snap.turn == snap.attacker:
        return ValidationResult(
            False, "перевод делает не защитник", "§5", Severity.CRITICAL)

    if not card_in_list(card, snap.my_hand):
        return ValidationResult(
            False, f"карты {card_str(card)} нет в руке", "§5",
            Severity.CRITICAL, {"card": card})

    # Верхняя непобитая атака — того же ранга.
    undefended = [p for p in snap.table if not p.get("defended")]
    if not undefended:
        return ValidationResult(
            False, "нет непобитой атаки для перевода", "§5", Severity.CRITICAL)

    top_rank = undefended[-1]["attack"]["r"]
    if card["r"] != top_rank:
        return ValidationResult(
            False,
            f"переводить можно картой ранга '{top_rank}' "
            f"(верхняя непобитая атака), а не '{card['r']}' (§5)",
            "§5", Severity.CRITICAL,
            {"card": card, "expected_rank": top_rank})

    # Ограничение перевода (§5): у нового защитника карт должно быть не меньше,
    # чем получится на столе после перевода.
    new_defender_hand = snap.opp_hand_count if snap.attacker == "me" else len(snap.my_hand)
    cards_after = len(undefended) + 1
    if new_defender_hand < cards_after:
        return ValidationResult(
            False,
            f"нельзя перевести: у нового защитника {new_defender_hand} карт(ы), "
            f"а после перевода потребуется отбить {cards_after} (§5)",
            "§5", Severity.CRITICAL,
            {"new_defender_hand": new_defender_hand, "cards_after": cards_after})

    # Лимит пар тоже должен соблюдаться.
    by_rules = snap.first_trick_limit if snap.first_trick else 6
    max_pairs = min(by_rules, new_defender_hand)
    if len(snap.table) >= max_pairs:
        return ValidationResult(
            False,
            f"превышен лимит пар при переводе ({len(snap.table)} ≥ {max_pairs}) (§5+§6)",
            "§5", Severity.CRITICAL,
            {"table_len": len(snap.table), "max_pairs": max_pairs})

    return ValidationResult(True, "", "§5", Severity.INFO)


def validate_take(snap: GameStateSnapshot) -> ValidationResult:
    """§7 — взять («беру»)."""
    if snap.phase != "defense":
        return ValidationResult(
            False, f"взять можно только в фазе защиты (сейчас '{snap.phase}')", "§7",
            Severity.CRITICAL)
    if snap.turn == snap.attacker:
        return ValidationResult(
            False, "берёт атакующий (должен защищающийся)", "§7", Severity.CRITICAL)
    if not snap.table:
        return ValidationResult(
            False, "нечего брать (стол пуст)", "§7", Severity.CRITICAL)
    return ValidationResult(True, "", "§7", Severity.INFO)


def validate_done(snap: GameStateSnapshot) -> ValidationResult:
    """§7 — «бито»."""
    if snap.phase != "attack":
        return ValidationResult(
            False, f"«бито» объявляется в фазе атаки (сейчас '{snap.phase}')", "§7",
            Severity.CRITICAL)
    if snap.turn != snap.attacker:
        return ValidationResult(
            False, "«бито» объявляет не атакующий", "§7", Severity.CRITICAL)
    if not snap.table:
        return ValidationResult(
            False, "стол пуст — нечего завершать", "§7", Severity.CRITICAL)
    undefended = [p for p in snap.table if not p.get("defended")]
    if undefended:
        return ValidationResult(
            False,
            f"есть {len(undefended)} непобитых атак — «бито» объявить нельзя (§7)",
            "§7", Severity.CRITICAL,
            {"undefended_count": len(undefended)})
    return ValidationResult(True, "", "§7", Severity.INFO)


def validate_pass(snap: GameStateSnapshot) -> ValidationResult:
    """Pass — отказ от подкидывания / завершение «вдогонку» (§7)."""
    if snap.phase not in ("attack", "pursue"):
        return ValidationResult(
            False, f"pass невозможен в фазе '{snap.phase}'", "§7",
            Severity.CRITICAL)
    if snap.turn != snap.attacker:
        return ValidationResult(
            False, "pass делает не атакующий", "§7", Severity.CRITICAL)
    if not snap.table:
        return ValidationResult(
            False, "стол пуст — нечего завершать", "§7", Severity.CRITICAL)
    if snap.phase == "attack":
        undefended = [p for p in snap.table if not p.get("defended")]
        if undefended:
            return ValidationResult(
                False,
                f"pass невозможен: есть {len(undefended)} непобитых атак "
                f"(нужно «бито» только после полной защиты, §7)",
                "§7", Severity.CRITICAL,
                {"undefended_count": len(undefended)})
    return ValidationResult(True, "", "§7", Severity.INFO)


# ===========================================================================
# Диспетчер валидации
# ===========================================================================

def validate_move(snap: GameStateSnapshot, move: dict) -> ValidationResult:
    """
    Универсальный валидатор хода.
    move = {
        "action": "attack|defend|transfer|toss|take|done|pass",
        "card":   {"r":..,"s":..} | None,
        "target": {"r":..,"s":..} | None,  # для defend
    }
    """
    action = move.get("action", "")
    card = move.get("card")
    target = move.get("target")

    if action in ("attack", "toss"):
        if card is None:
            return ValidationResult(False, "attack/toss требует card", "§4",
                                     Severity.CRITICAL)
        return validate_attack(snap, card)
    if action == "defend":
        if card is None:
            return ValidationResult(False, "defend требует card", "§4",
                                     Severity.CRITICAL)
        return validate_defend(snap, card, target)
    if action == "transfer":
        if card is None:
            return ValidationResult(False, "transfer требует card", "§5",
                                     Severity.CRITICAL)
        return validate_transfer(snap, card)
    if action == "take":
        return validate_take(snap)
    if action == "done":
        return validate_done(snap)
    if action == "pass":
        return validate_pass(snap)
    return ValidationResult(False, f"неизвестное действие '{action}'", "",
                             Severity.CRITICAL)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    print("=== Self-test rules_validator ===\n")

    # Тест 1: легальная атака пустым столом.
    snap = GameStateSnapshot(
        trump="spades", my_hand=[{"r": "7", "s": "clubs"}],
        opp_hand_count=6, table=[], attacker="me", turn="me", phase="attack",
    )
    r = validate_move(snap, {"action": "attack", "card": {"r": "7", "s": "clubs"}})
    assert r.ok, r.reason
    print(f"✓ Тест 1 (легальная атака): {r.reason or 'OK'}")

    # Тест 2: подкидывание ранга, которого нет на столе.
    snap = GameStateSnapshot(
        trump="spades",
        my_hand=[{"r": "K", "s": "hearts"}, {"r": "7", "s": "diamonds"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "7", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="me", turn="me", phase="attack",
    )
    r = validate_move(snap, {"action": "toss", "card": {"r": "K", "s": "hearts"}})
    assert not r.ok and r.severity == Severity.CRITICAL
    print(f"✓ Тест 2 (подкид K при 7 на столе): {r.reason}")

    # Тест 3: защита некозырной старшей.
    snap = GameStateSnapshot(
        trump="hearts",
        my_hand=[{"r": "8", "s": "clubs"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "7", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    r = validate_move(snap, {"action": "defend", "card": {"r": "8", "s": "clubs"}})
    assert r.ok, r.reason
    print(f"✓ Тест 3 (8♣ бьёт 7♣): {r.reason or 'OK'}")

    # Тест 4: защита козырем некозырной атаки.
    snap = GameStateSnapshot(
        trump="hearts",
        my_hand=[{"r": "6", "s": "hearts"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "A", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    r = validate_move(snap, {"action": "defend", "card": {"r": "6", "s": "hearts"}})
    assert r.ok, r.reason
    print(f"✓ Тест 4 (6♥ бьёт A♣ козырем): {r.reason or 'OK'}")

    # Тест 5: попытка перевода с недостатком карт у нового защитника.
    snap = GameStateSnapshot(
        trump="spades", transfer_enabled=True,
        my_hand=[{"r": "8", "s": "hearts"}],
        opp_hand_count=1,  # у соперника (бывшего атакующего) всего 1 карта
        table=[{"attack": {"r": "8", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    r = validate_move(snap, {"action": "transfer", "card": {"r": "8", "s": "hearts"}})
    assert not r.ok and "нельзя перевести" in r.reason.lower()
    print(f"✓ Тест 5 (перевод при 1 карте у соперника): {r.reason}")

    # Тест 6: «бито» с непобитой атакой.
    snap = GameStateSnapshot(
        trump="spades",
        my_hand=[{"r": "7", "s": "clubs"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "7", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="me", turn="me", phase="attack",
    )
    r = validate_move(snap, {"action": "done"})
    assert not r.ok and "непобитых" in r.reason
    print(f"✓ Тест 6 («бито» с непобитой атакой): {r.reason}")

    # Тест 7: Pass в фазе defense с непобитыми атаками — критический штраф.
    snap = GameStateSnapshot(
        trump="spades",
        my_hand=[{"r": "8", "s": "clubs"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "7", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    r = validate_move(snap, {"action": "pass"})
    assert not r.ok
    print(f"✓ Тест 7 (pass в защите): {r.reason}")

    print("\nВсе тесты пройдены.")
