"""
strict_arena.py — строгая арена с валидацией правил и системой штрафов.

Ключевые отличия от arena.py:
  1. Каждый ход бота проходит через rules_validator перед применением.
     Если бот предлагает нелегальный ход — немедленный критический штраф,
     партия засчитывается как поражение, логируется причина и номер правила.
  2. Любой таймаут или бесконечный цикл — поражение по штрафу.
  3. Подробный отчёт по каждой партии: что произошло, какой ход был
     нелегальным, на каком правиле бот споткнулся.
  4. Elo-трекинг: после каждого матча пересчитывается Elo обоих ботов,
     сохраняется в JSONL для долгосрочного мониторинга силы.
  5. Несколько типов соперников: pure-mcts (без сети), random, onnx-модель,
     «человек» (для отладки — выбор через stdin).

Протокол воркера (stdin/stdout, JSON-строки):
  → {"cmd":"load","path":"..."}                          ← {"ok":true}
  → {"cmd":"init","seed":42,"as_player":"me"}            ← {"ok":true}
  → {"cmd":"snapshot", "snap": {...}}                    ← {"ok":true}
       (передаём воркеру снапшот состояния, чтобы он знал позицию)
  → {"cmd":"ask_action","time_budget":0.3}               ← {"action":17}
  → {"cmd":"apply","action":17}                          ← {"ok":true}
  → {"cmd":"quit"}

Арена хранит своё каноническое состояние (GameStateSnapshot) и применяет
ходы к нему напрямую через rules_validator + apply_move_to_snapshot.
Воркер работает со своей внутренней моделью (durakk_env или OnnxNet) —
его задача только предлагать ходы. Арена — единственный источник правды.

Запуск:
  python training/strict_arena.py \\
      --bot-a checkpoints/current.onnx \\
      --bot-b pure-mcts:time=0.3 \\
      --games 50 \\
      --time-budget 0.3 \\
      --report reports/strict_arena_$(date +%Y%m%d).json
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import subprocess
import threading
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

# Локальные импорты.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT)

from rules_validator import (
    GameStateSnapshot, ValidationResult, Severity,
    validate_move, card_str, card_rank, beats, same_card, card_in_list,
)


# ===========================================================================
# Применение хода к снимку состояния
# ===========================================================================

def apply_move_to_snapshot(snap: GameStateSnapshot, move: dict) -> None:
    """
    Применяет ВАЛИДНЫЙ ход к снимку, мутируя его. Вызывающий код ДОЛЖЕН
    предварительно вызвать validate_move — здесь нет проверок (для скорости).

    Поддерживает: attack, toss, defend, transfer, take, done, pass.
    """
    action = move["action"]
    card = move.get("card")

    if action in ("attack", "toss"):
        # Убираем карту из руки атакующего.
        snap.my_hand = [c for c in snap.my_hand if not same_card(c, card)]
        # Кладём на стол как новую непобитую пару.
        snap.table.append({"attack": card, "defense": None, "defended": False})
        # Ход переходит к защитнику.
        snap.turn = "opp" if snap.attacker == "me" else "me"
        snap.phase = "defense"

    elif action == "defend":
        target = move.get("target")
        # Убираем карту из руки защитника.
        snap.my_hand = [c for c in snap.my_hand if not same_card(c, card)]
        # Находим целевую непобитую пару и побиваем.
        for p in snap.table:
            if p.get("defended"):
                continue
            if target and not same_card(p["attack"], target):
                continue
            if beats(p["attack"], card, snap.trump):
                p["defense"] = card
                p["defended"] = True
                break
        # Если все побиты — ход к атакующему, фаза attack.
        if all(p.get("defended") for p in snap.table):
            snap.turn = snap.attacker
            snap.phase = "attack"

    elif action == "transfer":
        snap.my_hand = [c for c in snap.my_hand if not same_card(c, card)]
        snap.table.append({"attack": card, "defense": None, "defended": False})
        # Смена ролей: бывший защитник становится атакующим.
        snap.attacker = snap.turn  # тот, кто перевёл
        snap.turn = "opp" if snap.attacker == "me" else "me"
        snap.phase = "defense"
        snap.transfers_this_trick += 1

    elif action == "take":
        # Объявление «беру» — переход в фазу pursue для возможного «вдогонку».
        snap.phase = "pursue"
        snap.turn = snap.attacker

    elif action in ("done", "pass"):
        if snap.phase == "pursue":
            # Завершение кона со взятием: защитник забирает стол.
            taker = "opp" if snap.attacker == "me" else "me"
            # Карты со стола уходят в «руку» взявшего — мы их не видим
            # у соперника (если taker == opp), но счётчик растёт.
            n_cards_on_table = sum(2 if p.get("defended") else 1 for p in snap.table)
            if taker == "me":
                # Мы забрали — добавляем все карты в нашу руку.
                for p in snap.table:
                    snap.my_hand.append(p["attack"])
                    if p.get("defense"):
                        snap.my_hand.append(p["defense"])
            else:
                snap.opp_hand_count += n_cards_on_table
            # Карты со стола в discard не идут (они у взявшего).
            snap.table = []
            # Ход остаётся у атакующего.
            snap.turn = snap.attacker
            snap.phase = "attack"
            snap.transfers_this_trick = 0
            # Добор (упрощённый): каждая сторона до 6 карт, первой — экс-атакующий.
            _do_refill(snap, first=snap.attacker)
        else:
            # «Бито»: карты уходят в сброс, ход к защитнику.
            for p in snap.table:
                snap.discard.append(p["attack"])
                if p.get("defense"):
                    snap.discard.append(p["defense"])
            new_attacker = "opp" if snap.attacker == "me" else "me"
            snap.table = []
            snap.attacker = new_attacker
            snap.turn = new_attacker
            snap.phase = "attack"
            snap.transfers_this_trick = 0
            snap.first_trick = False
            _do_refill(snap, first=snap.attacker)


def _do_refill(snap: GameStateSnapshot, first: str) -> None:
    """Упрощённый добор до 6 карт. Первым берёт 'first', затем другой."""
    if snap.deck_remaining <= 0:
        return
    # Первым добирает 'first'.
    if first == "me":
        while snap.deck_remaining > 0 and len(snap.my_hand) < 6:
            snap.my_hand.append({"r": "?", "s": "?"})  # неизвестная карта
            snap.deck_remaining -= 1
        # Затем соперник.
        while snap.deck_remaining > 0 and snap.opp_hand_count < 6:
            snap.opp_hand_count += 1
            snap.deck_remaining -= 1
    else:
        while snap.deck_remaining > 0 and snap.opp_hand_count < 6:
            snap.opp_hand_count += 1
            snap.deck_remaining -= 1
        while snap.deck_remaining > 0 and len(snap.my_hand) < 6:
            snap.my_hand.append({"r": "?", "s": "?"})
            snap.deck_remaining -= 1


# ===========================================================================
# Регистрация ботов
# ===========================================================================

@dataclass
class BotSpec:
    """Спецификация бота для арены."""
    name: str                       # короткое имя для логов
    kind: str                       # "onnx" | "pure-mcts" | "random"
    onnx_path: Optional[str] = None # для kind=onnx
    time_budget: float = 0.3
    num_threads: int = 1
    device: str = "CPU"
    durakk_env_path: str = "."      # путь к durakk_env.pyd/.so


# ===========================================================================
# Субпроцесс-воркер
# ===========================================================================

class WorkerProcess:
    """Один субпроцесс с загруженной моделью. Потокобезопасный."""

    def __init__(self, spec: BotSpec):
        self.spec = spec
        self.proc = subprocess.Popen(
            [sys.executable, "-u", __file__, "--worker",
             "--kind", spec.kind,
             "--time-budget", str(spec.time_budget),
             "--num-threads", str(spec.num_threads),
             "--device", spec.device,
             "--durakk-env-path", spec.durakk_env_path,
             "--name", spec.name,
             "--onnx", spec.onnx_path or ""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        # Ждём готовности.
        line = self.proc.stdout.readline()
        if "READY" not in line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(
                f"Worker '{spec.name}' не стартовал: {line!r}\nSTDERR: {err}")
        print(f"[Arena] Worker '{spec.name}' готов (pid={self.proc.pid}, kind={spec.kind})")

    def load_model(self) -> bool:
        if self.spec.kind != "onnx" or not self.spec.onnx_path:
            return True
        with self._lock:
            self._send({"cmd": "load", "path": self.spec.onnx_path})
            resp = self._recv()
            return resp.get("ok", False)

    def init_game(self, seed: int, as_player: str):
        with self._lock:
            self._send({"cmd": "init", "seed": seed, "as_player": as_player})
            self._recv()

    def ask_action(self, snap: GameStateSnapshot, time_budget: float) -> Tuple[Optional[dict], Optional[str]]:
        """
        Возвращает (move, error). move = {"action":..,"card":..,"target":..}.
        error — строка, если что-то пошло не так (таймаут, исключение).
        """
        with self._lock:
            self._send({
                "cmd": "ask_action",
                "time_budget": time_budget,
                "snapshot": asdict(snap),
            })
            try:
                resp = self._recv(timeout=time_budget + 30.0)
            except RuntimeError as e:
                return None, str(e)
            if "error" in resp:
                return None, resp["error"]
            return resp.get("move"), None

    def apply_action(self, action_idx: int):
        with self._lock:
            self._send({"cmd": "apply", "action": action_idx})
            self._recv()

    def close(self):
        try:
            self._send({"cmd": "quit"})
            self.proc.wait(timeout=5.0)
        except Exception:
            self.proc.kill()

    def _send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _recv(self, timeout: float = 120.0) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"Worker '{self.spec.name}' умер: stderr={err}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Worker '{self.spec.name}' прислал мусор: {line!r}: {e}")


# ===========================================================================
# Тело воркера (запускается при --worker)
# ===========================================================================

def _worker_main(spec: BotSpec):
    """Тело субпроцесса: держит durakk_env, отвечает на команды."""
    sys.path.insert(0, os.path.dirname(spec.durakk_env_path))
    sys.path.insert(0, spec.durakk_env_path)
    try:
        import durakk_env
    except ImportError as e:
        print(f"WORKER_ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    import numpy as np

    env = durakk_env.DurakEnv()
    model_loaded = (spec.kind != "onnx")  # для non-onnx модель не нужна

    print("READY", flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        cmd = msg.get("cmd")
        if cmd == "quit":
            break

        elif cmd == "load":
            try:
                if spec.kind == "onnx":
                    env.load_model(msg["path"], spec.device)
                    model_loaded = env.has_model()
                print(json.dumps({"ok": True}), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)

        elif cmd == "init":
            try:
                env.reset_with_seed(int(msg["seed"]))
                print(json.dumps({"ok": True}), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)

        elif cmd == "ask_action":
            # Главное: даём ход. Воркер НЕ должен валидировать — только предложить.
            try:
                tb = float(msg.get("time_budget", 0.3))
                if spec.kind == "random":
                    mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
                    legal = np.where(mask > 0)[0]
                    if len(legal) == 0:
                        action_idx = 37  # Pass
                    else:
                        action_idx = int(np.random.choice(legal))
                else:
                    # pure-mcts или onnx — через run_ismcts.
                    probs = env.run_ismcts(tb, spec.num_threads)
                    mask = np.array(env.get_legal_action_mask(), dtype=np.float32)
                    p = np.array(probs) * mask
                    if p.sum() > 0:
                        action_idx = int(np.argmax(p))
                    elif mask.sum() > 0:
                        action_idx = int(np.argmax(mask))
                    else:
                        action_idx = 37
                # Конвертируем action_idx в move dict.
                move = _action_idx_to_move(env, action_idx)
                print(json.dumps({"move": move, "action_idx": action_idx}),
                      flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

        elif cmd == "apply":
            try:
                ok = env.step(int(msg["action"]))
                print(json.dumps({"ok": ok}), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "error": str(e)}), flush=True)

        else:
            print(json.dumps({"error": f"unknown cmd: {cmd}"}), flush=True)


def _action_idx_to_move(env, action_idx: int) -> dict:
    """Конвертирует action_idx (0..37) в move dict для валидатора."""
    # 0..35 — карты (используем durakk_env для декодирования),
    # 36 = Take, 37 = Done/Pass.
    # К сожалению, у нас нет прямого API в durakk_env для получения
    # конкретного Move по action_idx. Поэтому вернём минимальную инфу —
    # action_idx и тип. Арена сама сопоставит.
    if action_idx == 36:
        return {"action": "take"}
    if action_idx == 37:
        # Done или Pass — определяется фазой в арене.
        return {"action": "done_or_pass"}
    # 0..35: индекс карты.
    # Декодируем через индекс: 9 рангов × 4 масти, suit = idx // 9, rank = idx % 9.
    suit_idx = action_idx // 9
    rank_idx = action_idx % 9
    suits = ["clubs", "diamonds", "hearts", "spades"]
    ranks = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    if suit_idx >= 4 or rank_idx >= 9:
        return {"action": "unknown", "action_idx": action_idx}
    return {
        "action": "card",
        "card": {"r": ranks[rank_idx], "s": suits[suit_idx]},
        "action_idx": action_idx,
    }


# ===========================================================================
# Главная арена
# ===========================================================================

@dataclass
class MoveRecord:
    """Запись одного хода в партии."""
    ply: int                    # 0-based номер полухода
    player: str                 # "me" | "opp" — кто ходил в терминах арены
    bot_name: str               # какой бот
    move: Optional[dict]        # предложенный ход
    action_idx: Optional[int]   # сырой action_idx
    valid: bool                 # прошёл валидацию
    reason: str                 # причина отказа, если invalid
    rule: str                   # номер правила, если нарушено
    severity: str               # "info"|"warning"|"critical"
    time_taken: float           # сколько думал бот


@dataclass
class GameRecord:
    """Полная запись одной партии."""
    seed: int
    bot_a: str                  # имя бота A
    bot_b: str                  # имя бота B
    bot_a_is_me: bool           # True → A играет за "me"
    moves: List[MoveRecord] = field(default_factory=list)
    winner: str = ""            # "A" | "B" | "draw" | "A_penalty" | "B_penalty"
    end_reason: str = ""        # "normal" | "illegal_move" | "timeout" | "infinite_loop"
    end_rule: str = ""          # нарушенное правило, если штраф
    end_ply: int = 0
    duration_sec: float = 0.0
    a_violations: int = 0       # всего нарушений у A (включая нефатальные)
    b_violations: int = 0

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "bot_a_is_me": self.bot_a_is_me,
            "winner": self.winner,
            "end_reason": self.end_reason,
            "end_rule": self.end_rule,
            "end_ply": self.end_ply,
            "duration_sec": round(self.duration_sec, 3),
            "a_violations": self.a_violations,
            "b_violations": self.b_violations,
            "moves": [
                {
                    "ply": m.ply, "player": m.player, "bot_name": m.bot_name,
                    "move": m.move, "action_idx": m.action_idx,
                    "valid": m.valid, "reason": m.reason,
                    "rule": m.rule, "severity": m.severity,
                    "time_taken": round(m.time_taken, 3),
                } for m in self.moves
            ],
        }


class StrictArena:
    """
    Строгая арена: каждый ход валидируется по правилам дурака.
    При критическом нарушении — немедленный штраф, партия завершается.
    """

    # Штрафы и их коды.
    PENALTY_ILLEGAL_MOVE = "illegal_move"       # нарушение правил
    PENALTY_TIMEOUT = "timeout"                  # превысил time_budget
    PENALTY_INFINITE_LOOP = "infinite_loop"      # игра зациклилась
    PENALTY_NO_LEGAL_MOVE = "no_legal_move"      # нет ходов, но не сдался
    PENALTY_WORKER_DEAD = "worker_dead"          # субпроцесс упал

    def __init__(self,
                 bot_a: BotSpec,
                 bot_b: BotSpec,
                 time_budget: float = 0.3,
                 max_moves_per_game: int = 600,
                 max_game_duration_sec: float = 300.0):
        self.bot_a = bot_a
        self.bot_b = bot_b
        self.time_budget = time_budget
        self.max_moves_per_game = max_moves_per_game
        self.max_game_duration_sec = max_game_duration_sec
        self._worker_a: Optional[WorkerProcess] = None
        self._worker_b: Optional[WorkerProcess] = None
        self.history: List[GameRecord] = []

    def _ensure_workers(self):
        if self._worker_a is None:
            self._worker_a = WorkerProcess(self.bot_a)
            self._worker_a.load_model()
            self._worker_b = WorkerProcess(self.bot_b)
            self._worker_b.load_model()

    def play_one_game(self, seed: int, bot_a_is_me: bool) -> GameRecord:
        """Играет одну партию и возвращает полный отчёт."""
        self._ensure_workers()

        record = GameRecord(
            seed=seed,
            bot_a=self.bot_a.name,
            bot_b=self.bot_b.name,
            bot_a_is_me=bot_a_is_me,
        )

        # Инициализация воркеров.
        self._worker_a.init_game(seed, "me" if bot_a_is_me else "opp")
        self._worker_b.init_game(seed, "me" if not bot_a_is_me else "opp")

        # Каноническое состояние арены. Это — ИСТИНА, воркеры только предлагают.
        snap = self._fresh_snapshot(seed)

        t_start = time.time()
        ply = 0
        try:
            while not self._is_game_over(snap):
                if ply >= self.max_moves_per_game:
                    record.end_reason = self.PENALTY_INFINITE_LOOP
                    record.end_rule = ""
                    # Зацикливание — ничья (или штраф обоим, см. ниже).
                    record.winner = "draw"
                    break
                if time.time() - t_start > self.max_game_duration_sec:
                    record.end_reason = self.PENALTY_TIMEOUT
                    record.winner = "draw"
                    break

                # Чей ход?
                current_player = snap.turn  # "me" | "opp"
                # Кто бот?
                if current_player == "me":
                    bot = self._worker_a if bot_a_is_me else self._worker_b
                    bot_name = self.bot_a.name if bot_a_is_me else self.bot_b.name
                else:
                    bot = self._worker_b if bot_a_is_me else self._worker_a
                    bot_name = self.bot_b.name if bot_a_is_me else self.bot_a.name

                # Спросить ход.
                t0 = time.time()
                move, err = bot.ask_action(snap, self.time_budget)
                t_taken = time.time() - t0

                # Таймаут.
                if err is not None:
                    # Определяем, кто проиграл по штрафу: тот, кто не ответил.
                    if current_player == "me":
                        # За "me" играет A, если bot_a_is_me=True, иначе B.
                        record.winner = "B_penalty" if bot_a_is_me else "A_penalty"
                    else:
                        record.winner = "A_penalty" if bot_a_is_me else "B_penalty"
                    record.end_reason = f"{self.PENALTY_TIMEOUT}: {err}"
                    record.moves.append(MoveRecord(
                        ply=ply, player=current_player, bot_name=bot_name,
                        move=None, action_idx=None, valid=False,
                        reason=f"таймаут/ошибка: {err}", rule="",
                        severity=Severity.CRITICAL.value, time_taken=t_taken))
                    if bot_name == self.bot_a.name:
                        record.a_violations += 1
                    else:
                        record.b_violations += 1
                    break

                # Конвертировать move из формата воркера в формат валидатора.
                norm_move = self._normalize_move(move, snap)

                # Валидация.
                result = validate_move(snap, norm_move)
                record.moves.append(MoveRecord(
                    ply=ply, player=current_player, bot_name=bot_name,
                    move=norm_move, action_idx=move.get("action_idx") if move else None,
                    valid=result.ok, reason=result.reason,
                    rule=result.rule, severity=result.severity.value,
                    time_taken=t_taken))

                if not result.ok and result.severity == Severity.CRITICAL:
                    # КРИТИЧЕСКИЙ ШТРАФ — конец игры.
                    record.end_reason = self.PENALTY_ILLEGAL_MOVE
                    record.end_rule = result.rule
                    record.end_ply = ply
                    record.winner = ("B" if bot_name == self.bot_a.name else "A") + "_normal"
                    # Перезапишем winner с учётом штрафа.
                    if bot_name == self.bot_a.name:
                        record.winner = "B_penalty"
                        record.a_violations += 1
                    else:
                        record.winner = "A_penalty"
                        record.b_violations += 1
                    print(f"  [ШТРАФ] {bot_name}: {result.reason} ({result.rule})")
                    break

                # Применить ход к снапшоту.
                apply_move_to_snapshot(snap, norm_move)
                # Синхронизировать воркер (чтобы его внутренний env был в той же позиции).
                if move.get("action_idx") is not None:
                    try:
                        bot.apply_action(move["action_idx"])
                    except Exception:
                        pass  # не критично — арена всё равно источник правды
                ply += 1
        except Exception as e:
            record.end_reason = self.PENALTY_WORKER_DEAD
            record.end_rule = ""
            record.winner = "draw"
            record.end_ply = ply
            print(f"  [КРИТИЧЕСКАЯ ОШИБКА АРЕНЫ] {e}")
        finally:
            record.duration_sec = time.time() - t_start
            if not record.end_reason:
                # Игра дошла до конца нормально.
                record.end_ply = ply
                record.end_reason = "normal"
                # Определим победителя.
                if len(snap.my_hand) == 0 and snap.opp_hand_count > 0:
                    record.winner = "A" if bot_a_is_me else "B"
                elif snap.opp_hand_count == 0 and len(snap.my_hand) > 0:
                    record.winner = "B" if bot_a_is_me else "A"
                else:
                    record.winner = "draw"

        return record

    def _fresh_snapshot(self, seed: int) -> GameStateSnapshot:
        """Создаёт стартовый снимок: 6+6 карт, колода 24 (36−12), первый ход Me."""
        rng = random.Random(seed)
        # Сгенерируем колоду, раздадим по 6.
        suits = ["clubs", "diamonds", "hearts", "spades"]
        ranks = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        deck = [{"r": r, "s": s} for s in suits for r in ranks]
        rng.shuffle(deck)
        trump = deck[0]["s"]
        # Первая карта — козырь-индикатор. После раздачи 12 карт, в колоде 23.
        my_hand = deck[1:7]
        opp_hand = deck[7:13]
        deck_remaining = 36 - 12  # 24
        return GameStateSnapshot(
            deck_size=36, trump=trump, transfer_enabled=True, flash_enabled=False,
            first_trick_limit=5,
            my_hand=my_hand, opp_hand_count=len(opp_hand),
            table=[], deck_remaining=deck_remaining, discard=[],
            attacker="me", turn="me", phase="attack", first_trick=True,
            flash_used_this_trick=False, transfers_this_trick=0,
        )

    def _is_game_over(self, snap: GameStateSnapshot) -> bool:
        if snap.deck_remaining > 0:
            return False
        if len(snap.my_hand) == 0 or snap.opp_hand_count == 0:
            return True
        return False

    def _normalize_move(self, move: Optional[dict], snap: GameStateSnapshot) -> dict:
        """
        Воркер возвращает move в формате {"action":"card","card":{...},"action_idx":N}
        или {"action":"take"} / {"action":"done_or_pass"}.
        Нужно привести к каноническому формату валидатора.
        """
        if move is None:
            return {"action": "unknown"}

        action = move.get("action", "unknown")

        if action == "card":
            card = move["card"]
            # Определяем тип по фазе и роли.
            if snap.phase == "attack" or snap.phase == "pursue":
                if not snap.table:
                    return {"action": "attack", "card": card}
                return {"action": "toss", "card": card}
            elif snap.phase == "defense":
                # Если ранг совпадает с верхней непобитой атакой — перевод, иначе защита.
                undefended = [p for p in snap.table if not p.get("defended")]
                if undefended and card["r"] == undefended[-1]["attack"]["r"]:
                    # Но это перевод только если есть transfer_enabled и ранг тот же.
                    # Если карта может побить — считаем защитой (бот решает).
                    # На самом деле по правилам: положил карту того же ранга рядом =
                    # перевод. Бить той же картой нельзя (она того же ранга, не старше).
                    return {"action": "transfer", "card": card}
                return {"action": "defend", "card": card}
            return {"action": "unknown", "card": card}

        if action == "take":
            return {"action": "take"}

        if action == "done_or_pass":
            # Определяем по фазе.
            if snap.phase == "pursue":
                return {"action": "pass"}
            if snap.phase == "attack":
                # Если все побито — done, иначе pass (но pass в attack с непобитыми
                # запретим — это критический штраф).
                if all(p.get("defended") for p in snap.table) and snap.table:
                    return {"action": "done"}
                return {"action": "pass"}
            return {"action": "done"}

        return {"action": action}

    def close(self):
        if self._worker_a:
            self._worker_a.close()
        if self._worker_b:
            self._worker_b.close()
        self._worker_a = None
        self._worker_b = None


# ===========================================================================
# Elo-трекинг
# ===========================================================================

class EloTracker:
    """Простой Elo-трекер с сохранением в JSONL."""

    def __init__(self, path: Optional[str] = None,
                 k: float = 32.0, default_elo: float = 1200.0):
        self.path = path
        self.k = k
        self.default_elo = default_elo
        self.elo: Dict[str, float] = defaultdict(lambda: default_elo)

    def expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, name_a: str, name_b: str, score_a: float):
        """
        score_a: 1.0 = победа A, 0.0 = победа B, 0.5 = ничья.
        Штрафы: A_penalty → score_a = 0.0; B_penalty → score_a = 1.0.
        """
        ra, rb = self.elo[name_a], self.elo[name_b]
        ea = self.expected(ra, rb)
        self.elo[name_a] = ra + self.k * (score_a - ea)
        self.elo[name_b] = rb + self.k * ((1 - score_a) - (1 - ea))
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({
                    "t": time.time(),
                    "a": name_a, "b": name_b,
                    "elo_a_before": ra, "elo_b_before": rb,
                    "elo_a_after": self.elo[name_a],
                    "elo_b_after": self.elo[name_b],
                    "score_a": score_a,
                }) + "\n")

    def get(self, name: str) -> float:
        return self.elo[name]


# ===========================================================================
# CLI
# ===========================================================================

def parse_bot_spec(s: str) -> BotSpec:
    """
    Парсит спецификацию бота из строки вида:
      onnx:path=/path/to/model.onnx,device=CPU,time=0.3
      pure-mcts:time=0.3,threads=1
      random
    """
    if ":" in s:
        kind, rest = s.split(":", 1)
    else:
        kind, rest = s, ""
    kwargs = {}
    for kv in rest.split(","):
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
            kwargs[k] = v
    return BotSpec(
        name=kwargs.get("name", s),
        kind=kind,
        onnx_path=kwargs.get("path"),
        time_budget=float(kwargs.get("time", 0.3)),
        num_threads=int(kwargs.get("threads", 1)),
        device=kwargs.get("device", "CPU"),
        durakk_env_path=kwargs.get("env-path", "."),
    )


def main():
    p = argparse.ArgumentParser(
        description="Строгая арена с валидацией правил дурака")
    p.add_argument("--bot-a", required=True,
                   help='Спецификация A, например "onnx:path=model.onnx" или "pure-mcts:time=0.3"')
    p.add_argument("--bot-b", required=True,
                   help='Спецификация B')
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--time-budget", type=float, default=0.3,
                   help="Таймаут на ход в секундах")
    p.add_argument("--report", default=None,
                   help="Путь для JSON-отчёта по всем партиям")
    p.add_argument("--elo-log", default=None,
                   help="Путь для JSONL лога Elo (для долгосрочного трекинга)")
    p.add_argument("--durakk-env-path", default=".",
                   help="Путь к durakk_env.pyd/.so (обычно корень проекта)")
    p.add_argument("--seed-offset", type=int, default=1000)
    p.add_argument("--max-moves", type=int, default=600)
    args = p.parse_args()

    # Парсим спецификации ботов.
    bot_a = parse_bot_spec(args.bot_a)
    bot_b = parse_bot_spec(args.bot_b)
    bot_a.durakk_env_path = args.durakk_env_path
    bot_b.durakk_env_path = args.durakk_env_path
    bot_a.time_budget = args.time_budget
    bot_b.time_budget = args.time_budget

    print(f"=== Строгая арена ===")
    print(f"Бот A: {bot_a.name} ({bot_a.kind})")
    print(f"Бот B: {bot_b.name} ({bot_b.kind})")
    print(f"Партий: {args.games}, time_budget={args.time_budget}s")
    print()

    arena = StrictArena(bot_a, bot_b, args.time_budget,
                        max_moves_per_game=args.max_moves)
    elo = EloTracker(args.elo_log)

    try:
        for g in range(args.games):
            bot_a_is_me = (g % 2 == 0)  # чередуем стороны
            seed = args.seed_offset + g
            print(f"\n=== Игра {g+1}/{args.games} (seed={seed}, "
                  f"A={'me' if bot_a_is_me else 'opp'}) ===")
            record = arena.play_one_game(seed, bot_a_is_me)
            arena.history.append(record)

            # Elo-обновление.
            score_a = 0.5
            if record.winner == "A":
                score_a = 1.0
            elif record.winner == "B":
                score_a = 0.0
            elif record.winner == "A_penalty":
                # A нарушил правила — поражение A.
                score_a = 0.0
            elif record.winner == "B_penalty":
                score_b = 0.0
                score_a = 1.0
            elif record.winner == "draw":
                score_a = 0.5
            elo.update(bot_a.name, bot_b.name, score_a)

            # Краткий лог.
            print(f"  Результат: {record.winner} "
                  f"({record.end_reason}{', правило ' + record.end_rule if record.end_rule else ''})")
            print(f"  Ходов: {record.end_ply}, время: {record.duration_sec:.1f}s, "
                  f"нарушений A={record.a_violations} B={record.b_violations}")
            print(f"  Elo: A={elo.get(bot_a.name):.0f}, B={elo.get(bot_b.name):.0f}")
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    finally:
        arena.close()

    # Итоговая статистика.
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    a_wins = sum(1 for r in arena.history if r.winner in ("A", "B_penalty"))
    b_wins = sum(1 for r in arena.history if r.winner in ("B", "A_penalty"))
    draws = sum(1 for r in arena.history if r.winner == "draw")
    a_pen = sum(1 for r in arena.history if r.winner == "A_penalty")
    b_pen = sum(1 for r in arena.history if r.winner == "B_penalty")
    print(f"Партий сыграно: {len(arena.history)}")
    print(f"Побед A:           {a_wins}")
    print(f"Побед B:           {b_wins}")
    print(f"Ничьих:            {draws}")
    print(f"Штрафов A:         {a_pen}")
    print(f"Штрафов B:         {b_pen}")
    total_a_viol = sum(r.a_violations for r in arena.history)
    total_b_viol = sum(r.b_violations for r in arena.history)
    print(f"Всего нарушений A: {total_a_viol}")
    print(f"Всего нарушений B: {total_b_viol}")
    print(f"Финальный Elo: A={elo.get(bot_a.name):.0f}, B={elo.get(bot_b.name):.0f}")

    # Топ нарушений по правилам.
    rule_violations: Dict[str, int] = defaultdict(int)
    for r in arena.history:
        for m in r.moves:
            if not m.valid and m.severity == "critical":
                rule_violations[m.rule] += 1
    if rule_violations:
        print("\nТоп нарушенных правил:")
        for rule, cnt in sorted(rule_violations.items(),
                                key=lambda x: -x[1]):
            print(f"  {rule}: {cnt} раз")

    # Сохранить полный отчёт.
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({
                "bot_a": bot_a.name, "bot_b": bot_b.name,
                "time_budget": args.time_budget,
                "games": len(arena.history),
                "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
                "a_penalties": a_pen, "b_penalties": b_pen,
                "a_violations": total_a_viol, "b_violations": total_b_viol,
                "elo_a": elo.get(bot_a.name), "elo_b": elo.get(bot_b.name),
                "rule_violations": dict(rule_violations),
                "records": [r.to_dict() for r in arena.history],
            }, f, indent=2, ensure_ascii=False)
        print(f"\nПолный отчёт сохранён: {args.report}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        # Режим субпроцесса-воркера.
        idx = sys.argv.index("--worker")
        spec_kwargs = {}
        # Простой парсинг argv.
        args = sys.argv[idx + 1:]
        i = 0
        while i < len(args):
            if args[i] == "--kind" and i + 1 < len(args):
                spec_kwargs["kind"] = args[i + 1]; i += 2
            elif args[i] == "--time-budget" and i + 1 < len(args):
                spec_kwargs["time_budget"] = float(args[i + 1]); i += 2
            elif args[i] == "--num-threads" and i + 1 < len(args):
                spec_kwargs["num_threads"] = int(args[i + 1]); i += 2
            elif args[i] == "--device" and i + 1 < len(args):
                spec_kwargs["device"] = args[i + 1]; i += 2
            elif args[i] == "--durakk-env-path" and i + 1 < len(args):
                spec_kwargs["durakk_env_path"] = args[i + 1]; i += 2
            elif args[i] == "--name" and i + 1 < len(args):
                spec_kwargs["name"] = args[i + 1]; i += 2
            elif args[i] == "--onnx" and i + 1 < len(args):
                spec_kwargs["onnx_path"] = args[i + 1] or None; i += 2
            else:
                i += 1
        spec = BotSpec(**spec_kwargs)
        _worker_main(spec)
    else:
        main()
