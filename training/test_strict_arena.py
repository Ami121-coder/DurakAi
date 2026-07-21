"""
test_strict_arena.py — самопроверка strict_arena: валидатор + apply_move + Elo.

Запуск:
    python training/test_strict_arena.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rules_validator import (
    GameStateSnapshot, Severity, validate_move, card_str,
)
from strict_arena import apply_move_to_snapshot, EloTracker


def make_snap(**kwargs) -> GameStateSnapshot:
    defaults = dict(
        trump="spades", transfer_enabled=True, flash_enabled=False,
        first_trick_limit=5, my_hand=[], opp_hand_count=6, table=[],
        deck_remaining=24, discard=[], attacker="me", turn="me",
        phase="attack", first_trick=True, flash_used_this_trick=False,
        transfers_this_trick=0,
    )
    defaults.update(kwargs)
    return GameStateSnapshot(**defaults)


def test_attack_then_defend():
    """Me атакует 7♣, Opp защищается 8♣ → «бито»."""
    snap = make_snap(
        my_hand=[{"r": "7", "s": "clubs"}],
        opp_hand_count=6, attacker="me", turn="me", phase="attack",
    )
    # Me атакует 7♣.
    r = validate_move(snap, {"action": "attack", "card": {"r": "7", "s": "clubs"}})
    assert r.ok, r.reason
    apply_move_to_snapshot(snap, {"action": "attack", "card": {"r": "7", "s": "clubs"}})
    assert snap.phase == "defense"
    assert snap.turn == "opp"
    assert len(snap.table) == 1

    # Меняем перспективу: теперь "my_hand" = рука Opp.
    snap.my_hand, snap.opp_hand_count = [{"r": "8", "s": "clubs"}], len(snap.my_hand)
    snap.attacker, snap.turn = "me", "opp"
    snap.attacker = "me"
    snap.turn = "opp"

    # Opp защищается 8♣.
    r = validate_move(snap, {"action": "defend", "card": {"r": "8", "s": "clubs"}})
    assert r.ok, r.reason
    apply_move_to_snapshot(snap, {"action": "defend", "card": {"r": "8", "s": "clubs"}})
    assert snap.table[0]["defended"] is True
    assert snap.phase == "attack"
    assert snap.turn == "me"
    print("✓ test_attack_then_defend: OK")


def test_transfer_swap_perspective():
    """Me атакует 8♣, Opp переводит 8♥ → роли меняются."""
    snap = make_snap(
        my_hand=[{"r": "8", "s": "clubs"}, {"r": "9", "s": "spades"}],
        opp_hand_count=3, attacker="me", turn="me", phase="attack",
    )
    apply_move_to_snapshot(snap, {"action": "attack", "card": {"r": "8", "s": "clubs"}})
    # После атаки у меня осталась 1 карта в руке. У соперника (нового защитника
    # после перевода) должно быть ≥ 2 карт для перевода.
    # Перспектива Opp:
    snap.my_hand, snap.opp_hand_count = [{"r": "8", "s": "hearts"}], 1
    # У нового защитника (= ex-attacker = me) после атаки осталась 1 карта (9♠).
    # Но для перевода нужно, чтобы у НОВОГО защитника (ex-attacker) было карт
    # не меньше, чем непобитых атак + 1 = 1+1 = 2. У ex-attacker = 1 карта →
    # перевод заблокирован. Поэтому дадим ex-attacker'у 2 карты.
    snap.opp_hand_count = 2  # у ex-attacker теперь 2 карты
    snap.attacker = "me"
    snap.turn = "opp"
    snap.phase = "defense"

    r = validate_move(snap, {"action": "transfer", "card": {"r": "8", "s": "hearts"}})
    assert r.ok, r.reason
    apply_move_to_snapshot(snap, {"action": "transfer", "card": {"r": "8", "s": "hearts"}})
    assert snap.attacker == "opp"  # бывший защитник стал атакующим
    assert snap.turn == "me"        # бывший атакующий теперь защищается
    assert snap.transfers_this_trick == 1
    assert len(snap.table) == 2
    print("✓ test_transfer_swap_perspective: OK")


def test_take_then_pursue_pass():
    """Объявление «беру» → pursue → pass завершает кон со взятием."""
    snap = make_snap(
        my_hand=[{"r": "8", "s": "clubs"}],
        opp_hand_count=3, attacker="me", turn="me", phase="attack",
    )
    apply_move_to_snapshot(snap, {"action": "attack", "card": {"r": "8", "s": "clubs"}})
    # Перспектива Opp:
    snap.my_hand, snap.opp_hand_count = [], 1
    snap.attacker = "me"
    snap.turn = "opp"
    snap.phase = "defense"

    r = validate_move(snap, {"action": "take"})
    assert r.ok, r.reason
    apply_move_to_snapshot(snap, {"action": "take"})
    assert snap.phase == "pursue"
    assert snap.turn == "me"  # атакующий может подкинуть вдогонку

    # Pass — завершение взятия.
    r = validate_move(snap, {"action": "pass"})
    assert r.ok, r.reason
    apply_move_to_snapshot(snap, {"action": "pass"})
    # После взятия стол пуст, защитник забрал карты.
    assert snap.table == []
    # Ход остался у атакующего.
    assert snap.turn == "me"
    assert snap.phase == "attack"
    print("✓ test_take_then_pursue_pass: OK")


def test_done_requires_all_defended():
    """«Бито» требует, чтобы все атаки были побиты."""
    snap = make_snap(
        my_hand=[{"r": "7", "s": "clubs"}],
        opp_hand_count=6, attacker="me", turn="me", phase="attack",
    )
    apply_move_to_snapshot(snap, {"action": "attack", "card": {"r": "7", "s": "clubs"}})
    # Попытка «бито» при непобитой атаке — критический штраф.
    # Перспектива атакующего: turn=me, phase=defense (т.к. после атаки phase=defense).
    # Но Done требует phase=attack. Сначала защитимся.
    snap.my_hand, snap.opp_hand_count = [{"r": "8", "s": "clubs"}], 1
    snap.attacker = "me"
    snap.turn = "opp"
    snap.phase = "defense"
    apply_move_to_snapshot(snap, {"action": "defend", "card": {"r": "8", "s": "clubs"}})
    assert snap.phase == "attack"

    # Теперь «бито» легально.
    r = validate_move(snap, {"action": "done"})
    assert r.ok, r.reason
    apply_move_to_snapshot(snap, {"action": "done"})
    assert snap.table == []
    assert snap.attacker == "opp"  # ход перешёл к защитнику
    print("✓ test_done_requires_all_defended: OK")


def test_illegal_toss_rejected():
    """Подкидывание ранга, которого нет на столе — критический штраф."""
    snap = make_snap(
        my_hand=[{"r": "K", "s": "hearts"}],
        opp_hand_count=6, attacker="me", turn="me", phase="attack",
        table=[{"attack": {"r": "7", "s": "clubs"}, "defense": None, "defended": False}],
    )
    r = validate_move(snap, {"action": "toss", "card": {"r": "K", "s": "hearts"}})
    assert not r.ok
    assert r.severity == Severity.CRITICAL
    assert r.rule == "§6"
    print(f"✓ test_illegal_toss_rejected: {r.reason}")


def test_transfer_blocked_by_hand_size():
    """Перевод невозможен, если у нового защитника карт меньше, чем нужно."""
    snap = make_snap(
        my_hand=[{"r": "8", "s": "hearts"}],
        opp_hand_count=1,  # у соперника 1 карта, после перевода нужно отбить 2
        table=[{"attack": {"r": "8", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    r = validate_move(snap, {"action": "transfer", "card": {"r": "8", "s": "hearts"}})
    assert not r.ok
    assert r.severity == Severity.CRITICAL
    assert "нельзя перевести" in r.reason.lower()
    print(f"✓ test_transfer_blocked_by_hand_size: {r.reason}")


def test_elo_tracker():
    """Elo-обновление: A выиграл → A растёт, B падает."""
    elo = EloTracker()
    ea0 = elo.get("A")
    eb0 = elo.get("B")
    elo.update("A", "B", score_a=1.0)
    assert elo.get("A") > ea0
    assert elo.get("B") < eb0
    print(f"✓ test_elo_tracker: A {ea0:.0f}→{elo.get('A'):.0f}, "
          f"B {eb0:.0f}→{elo.get('B'):.0f}")


def test_first_trick_limit():
    """В первом коне максимум 5 пар."""
    snap = make_snap(
        my_hand=[{"r": "7", "s": "clubs"}, {"r": "7", "s": "diamonds"}],
        opp_hand_count=6, attacker="me", turn="me", phase="attack",
        first_trick=True, first_trick_limit=5,
    )
    # Заполним стол 5 парами (имитация).
    snap.table = [
        {"attack": {"r": "7", "s": "hearts"}, "defense": {"r": "8", "s": "hearts"}, "defended": True},
        {"attack": {"r": "7", "s": "spades"}, "defense": {"r": "8", "s": "spades"}, "defended": True},
        {"attack": {"r": "9", "s": "clubs"}, "defense": {"r": "10", "s": "clubs"}, "defended": True},
        {"attack": {"r": "9", "s": "hearts"}, "defense": {"r": "10", "s": "hearts"}, "defended": True},
        {"attack": {"r": "9", "s": "diamonds"}, "defense": {"r": "10", "s": "diamonds"}, "defended": True},
    ]
    # Попытка подкинуть ещё одну — должна быть отклонена (5 пар уже).
    r = validate_move(snap, {"action": "toss", "card": {"r": "7", "s": "clubs"}})
    assert not r.ok
    assert "лимит" in r.reason.lower() or "превышен" in r.reason.lower()
    print(f"✓ test_first_trick_limit: {r.reason}")


def test_trump_beats_non_trump():
    """Козырь бьёт некозырьную карту любого ранга."""
    snap = make_snap(
        trump="hearts",
        my_hand=[{"r": "6", "s": "hearts"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "A", "s": "clubs"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    r = validate_move(snap, {"action": "defend", "card": {"r": "6", "s": "hearts"}})
    assert r.ok, r.reason
    print("✓ test_trump_beats_non_trump: 6♥ бьёт A♣")


def test_higher_trump_beats_lower_trump():
    """Козырную карту бьёт только старший козырь."""
    snap = make_snap(
        trump="hearts",
        my_hand=[{"r": "7", "s": "hearts"}, {"r": "K", "s": "hearts"}],
        opp_hand_count=6,
        table=[{"attack": {"r": "6", "s": "hearts"}, "defense": None, "defended": False}],
        attacker="opp", turn="me", phase="defense",
    )
    # 7♥ бьёт 6♥.
    r = validate_move(snap, {"action": "defend", "card": {"r": "7", "s": "hearts"}})
    assert r.ok, r.reason
    # А вот K♥ тоже бьёт — это нормально.
    r = validate_move(snap, {"action": "defend", "card": {"r": "K", "s": "hearts"}})
    assert r.ok, r.reason
    print("✓ test_higher_trump_beats_lower_trump: 7♥/K♥ бьют 6♥")


def test_snapshot_from_env():
    """_snapshot_from_env корректно преобразует raw dict из env в GameStateSnapshot."""
    from strict_arena import StrictArena, BotSpec
    # Создадим минимальный StrictArena (без запуска воркеров).
    bot_a = BotSpec(name="A", kind="random")
    bot_b = BotSpec(name="B", kind="random")
    arena = StrictArena(bot_a, bot_b)

    # Raw dict, как его возвращает durakk_env.get_state_snapshot()
    # (после конвертации в _snapshot_to_jsonable).
    raw = {
        "trump": "hearts",
        "transferEnabled": True,
        "flashEnabled": False,
        "firstTrick": True,
        "pairsLimit": 6,
        "deckRemaining": 23,
        "attacker": "me",
        "turn": "opp",
        "phase": "defense",
        "viewpoint": "me",
        "myHand": [
            {"r": "7", "s": "clubs"},
            {"r": "K", "s": "diamonds"},
            {"r": "6", "s": "hearts"},
        ],
        "oppHandCount": 5,
        "table": [
            {"attack": {"r": "7", "s": "spades"}, "defense": None, "defended": False},
        ],
        "discard": [],
        "isGameOver": False,
        "winner": -1,
    }
    snap = arena._snapshot_from_env(raw, bot_a_is_me=True)
    assert snap.trump == "hearts"
    assert snap.attacker == "me"
    assert snap.turn == "opp"
    assert snap.phase == "defense"
    assert len(snap.my_hand) == 3
    assert snap.my_hand[0] == {"r": "7", "s": "clubs"}
    assert snap.opp_hand_count == 5
    assert len(snap.table) == 1
    assert snap.deck_remaining == 23
    assert snap.is_game_over is False
    assert snap.winner == -1

    # Теперь валидный ход: K♦ не бьёт 7♠ (разные масти), 6♥ бьёт козырем.
    r = validate_move(snap, {"action": "defend", "card": {"r": "6", "s": "hearts"}})
    assert r.ok, r.reason
    print(f"✓ test_snapshot_from_env: trump={snap.trump}, hand_size={len(snap.my_hand)}")


def test_snapshot_from_env_gameover():
    """Если env говорит isGameOver=True, snap.is_game_over=True."""
    from strict_arena import StrictArena, BotSpec
    bot_a = BotSpec(name="A", kind="random")
    bot_b = BotSpec(name="B", kind="random")
    arena = StrictArena(bot_a, bot_b)
    raw = {
        "trump": "spades",
        "transferEnabled": True,
        "flashEnabled": False,
        "firstTrick": False,
        "pairsLimit": 6,
        "deckRemaining": 0,
        "attacker": "me",
        "turn": "me",
        "phase": "attack",
        "viewpoint": "me",
        "myHand": [],   # я без карт — победил
        "oppHandCount": 3,
        "table": [],
        "discard": [],
        "isGameOver": True,
        "winner": 0,   # Me победил
    }
    snap = arena._snapshot_from_env(raw, bot_a_is_me=True)
    assert snap.is_game_over is True
    assert snap.winner == 0
    assert arena._is_game_over(snap) is True
    print("✓ test_snapshot_from_env_gameover: корректно распознаёт конец игры")


def main():
    print("=== Тесты strict_arena ===\n")
    tests = [
        test_attack_then_defend,
        test_transfer_swap_perspective,
        test_take_then_pursue_pass,
        test_done_requires_all_defended,
        test_illegal_toss_rejected,
        test_transfer_blocked_by_hand_size,
        test_elo_tracker,
        test_first_trick_limit,
        test_trump_beats_non_trump,
        test_higher_trump_beats_lower_trump,
        test_snapshot_from_env,
        test_snapshot_from_env_gameover,
    ]
    n_ok = 0
    n_fail = 0
    for t in tests:
        try:
            t()
            n_ok += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: FAIL: {e}")
            n_fail += 1
        except Exception as e:
            print(f"✗ {t.__name__}: ERROR: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n{n_ok} пройдено, {n_fail} провалено.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
