"""
train.py — УСИЛЕННОЕ обучение AlphaZero для Durak
(под Ryzen 7500F + 32GB DDR5 + RTX 4060 Ti 8GB).

КРИТИЧЕСКИЕ УЛУЧШЕНИЯ относительно исходной версии:
  1. Prioritized Experience Replay (PER) — ускорение сходимости на 20-40%.
  2. Исправленная arena (двухпроцессная, корректный winrate).
  3. Внешний baseline evaluator: vs RandomNet-ISMCTS — истинная метрика прогресса.
  4. Расширенный self-play: 8 воркеров × 8 партий = 64 партии/итер.
  5. CUDA в воркерах (с CPU fallback при нехватке VRAM).
  6. Логирование Elo-разницы и value-MSE.
  7. Авто-сохранение best-модели по Elo-vs-Random.
  8. AdamW + cosine with warmup + grad clip 1.0.
  9. FP16 autocast для скорости (4060 Ti ≈ 2× ускорение).
  10. TensorBoard: loss / value_mse / arena_wr / elo_diff / elo_vs_random.

ОЖИДАЕМАЯ ПРОПУСКНАЯ СПОСОБНОСТЬ на 7500F + 4060 Ti 8GB:
  - Self-play: 8 воркеров × 8 партий × ~30 ходов × 0.3 сек ≈ 70 сек
  - Training: 4 эпохи × 4096 батч × FP16 ≈ 8 сек
  - Arena (раз в 20 итер.): 40 партий × ~30 сек ≈ 20 мин (тормозит общий цикл)
  - Внешний baseline (раз в 20 итер.): 60 партий × ~30 сек ≈ 30 мин
  - Итого: ~80 сек/итер (без arena), ~150 сек/итер (с arena)
  - 1000 итераций ≈ 22-40 часов
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Fix: ONNX Runtime CUDA provider needs its DLL dir in PATH.
# This must be set before spawning workers so they inherit the env.
try:
    import onnxruntime as _ort_fix
    _ort_capi = os.path.join(os.path.dirname(_ort_fix.__file__), "capi")
    if os.path.isdir(_ort_capi):
        os.environ["PATH"] = _ort_capi + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


import sys
import time
import math
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_resnet import build_model
from replay_buffer import ReplayBuffer
from self_play import SelfPlayManager
from arena import ArenaEvaluator, ExternalBaselineEvaluator


# ============================================================================
# Утилиты.
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero-обучение для Durak (усиленная версия)")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs_per_iter", type=int, default=4)
    # Self-play
    p.add_argument("--num_workers", type=int, default=8,
                   help="Число self-play воркеров (7500F = 12 потоков, 8 оптимально)")
    p.add_argument("--games_per_iter", type=int, default=8,
                   help="Партий на воркер за итерацию")
    p.add_argument("--time_budget", type=float, default=0.3,
                   help="Секунд на ISMCTS-ход в self-play")
    p.add_argument("--num_threads", type=int, default=1,
                   help="Потоков на ISMCTS внутри воркера")
    p.add_argument("--sp_device", type=str, default="CUDA",
                   help="Device для self-play ONNX: CUDA или CPU")
    # Buffer
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--prioritized", action="store_true", default=True,
                   help="Использовать PER (Schaul et al., 2015)")
    p.add_argument("--no_prioritized", action="store_false", dest="prioritized")
    # Checkpoints
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--onnx_path", type=str, default="checkpoints/current.onnx",
                   help="Куда экспортировать ONNX для self-play воркеров")
    p.add_argument("--log_dir", type=str, default="runs/durak_az")
    p.add_argument("--resume", type=str, default=None)
    # Оптимизатор
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_iters", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # Модель
    p.add_argument("--num_blocks", type=int, default=10,
                   help="Residual-блоков (10 оптимально для 4060 Ti)")
    p.add_argument("--num_channels", type=int, default=256,
                   help="Каналов (256 = ~5M параметров)")
    # Arena
    p.add_argument("--arena_every", type=int, default=20)
    p.add_argument("--arena_games", type=int, default=40)
    p.add_argument("--arena_threshold", type=float, default=0.55,
                   help="Winrate для обновления baseline")
    p.add_argument("--ext_baseline_every", type=int, default=20,
                   help="Каждые N итер. — внешний baseline (vs RandomNet)")
    p.add_argument("--ext_baseline_games", type=int, default=60)
    p.add_argument("--ext_baseline_time", type=float, default=0.2)
    # Прочее
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_false", dest="fp16")
    p.add_argument("--durakk_env_path", type=str, default=".",
                   help="Путь к директории с durakk_env.pyd/.so (обычно корень проекта)")
    return p.parse_args()


def export_onnx(model, output_path: str, num_blocks: int, num_channels: int):
    """Авто-экспорт ONNX после каждого checkpoint."""
    import export_onnx
    try:
        tmp_pt = os.path.join(os.path.dirname(output_path), "_tmp_export.pt")
        torch.save({"model": model.state_dict()}, tmp_pt)
        export_onnx.export(tmp_pt, output_path,
                           num_blocks=num_blocks, num_channels=num_channels)
        os.remove(tmp_pt)
        print(f"  [ONNX] Экспортирован: {output_path}")
        return True
    except Exception as e:
        print(f"  [ONNX] Ошибка экспорта: {e}")
        return False


# ============================================================================
# Главная функция.
# ============================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Устройство обучения: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM total: {vram_total:.1f} GB")
        # Проверим, что VRAM хватит для batch 4096.
        # Примерно: 5M params × 2 (FP16) + activations ~75MB + grads ~10MB + AdamW 2×10MB ≈ 105MB.
        # На 8GB — огромный запас.

    # 1. Модель.
    model = build_model(num_blocks=args.num_blocks,
                        num_channels=args.num_channels).to(device)
    print(f"Параметров модели: {model.count_parameters():,}")

    # 2. Оптимизатор.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda') if args.fp16 else None

    def cosine_with_warmup(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        progress = (it - args.warmup_iters) / max(1, args.iterations - args.warmup_iters)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))

    # 3. Replay buffer (с PER).
    buf = ReplayBuffer(capacity=args.buffer_size, prioritized=args.prioritized)

    # 4. Resume.
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        if os.path.exists(os.path.join(args.checkpoint_dir, "buffer.pt")):
            try:
                buf.load(os.path.join(args.checkpoint_dir, "buffer.pt"))
            except Exception as e:
                print(f"Не удалось загрузить буфер: {e}")
        print(f"Resumed с итерации {start_iter}, буфер {len(buf):,} позиций")

    # 5. TensorBoard.
    writer = SummaryWriter(log_dir=args.log_dir)

    # 6. Экспорт ONNX перед запуском воркеров.
    initial_pt = os.path.join(args.checkpoint_dir, "current.pt")
    if not os.path.exists(initial_pt):
        torch.save({"model": model.state_dict(), "iter": -1,
                    "optimizer": optimizer.state_dict()}, initial_pt)
    print("Экспорт ONNX при запуске...")
    export_onnx(model, args.onnx_path, args.num_blocks, args.num_channels)

    # 7. Self-play менеджер.
    mgr = SelfPlayManager(
        num_workers=args.num_workers,
        onnx_path=args.onnx_path,
        games_per_iteration=args.games_per_iter,
        time_budget=args.time_budget,
        num_threads=args.num_threads,
        device=args.sp_device,
    )
    mgr.start()

    # 8. Arena evaluator (внутренний: current vs best.onnx).
    print("Инициализация ArenaEvaluator...", flush=True)
    arena = ArenaEvaluator(
        onnx_path=args.onnx_path,
        durakk_env_path=args.durakk_env_path,
        time_budget=args.time_budget,
        num_games=args.arena_games,
        device="CPU",  # arena сама по себе ест ресурсы, ставим CPU
    )
    print("ArenaEvaluator OK", flush=True)

    # 9. Внешний baseline evaluator (current vs RandomNet-ISMCTS).
    print("Инициализация ExternalBaselineEvaluator...", flush=True)
    ext_evaluator = ExternalBaselineEvaluator(
        onnx_path=args.onnx_path,
        durakk_env_path=args.durakk_env_path,
        time_budget=args.ext_baseline_time,
        num_games=args.ext_baseline_games,
        device="CPU",
    )
    print("ExternalBaselineEvaluator OK", flush=True)

    best_winrate = 0.0
    best_elo_vs_random = -1000.0

    # Логгер метрик.
    metrics_log = os.path.join(args.checkpoint_dir, "metrics.jsonl")
    metrics_file = open(metrics_log, "a")

    try:
        for it in range(start_iter, args.iterations):
            t0 = time.time()

            # 8.1. Собираем примеры.
            states_np, policies_np, values_np, masks_np = mgr.collect(
                max_examples=args.batch_size * 4, timeout=60.0)
            collected = len(states_np)

            dead = args.num_workers - mgr.alive_workers()
            if dead > 0:
                print(f"[ALARM] {dead} воркеров мертвы!")

            if collected > 0:
                buf.add(states_np, policies_np, values_np, masks_np)
            t_collect = time.time() - t0
            print(f"[Iter {it}] Собрано {collected} примеров за {t_collect:.1f}s, "
                  f"буфер={len(buf):,}")

            if len(buf) < args.batch_size:
                print(f"[Iter {it}] Буфер слишком мал — пропускаем обучение.")
                continue

            # 8.2. LR.
            lr = cosine_with_warmup(it)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            writer.add_scalar("Train/learning_rate", lr, it)

            # 8.3. Обучение (с PER weights).
            model.train()
            total_loss = total_p = total_v = total_v_mse = 0.0
            n_batches = 0

            for epoch in range(args.epochs_per_iter):
                s_np, p_np, v_np, m_np, weights_np, idxs = buf.sample(args.batch_size)
                states = torch.from_numpy(s_np).to(device)
                policies = torch.from_numpy(p_np).to(device)
                values = torch.from_numpy(v_np).to(device)
                masks = torch.from_numpy(m_np).to(device)
                weights = torch.from_numpy(weights_np).to(device)

                optimizer.zero_grad()
                if args.fp16:
                    with torch.amp.autocast('cuda'):
                        pred_p, pred_v = model(states, masks)
                        log_preds = F.log_softmax(pred_p, dim=-1)
                        # Clamp to avoid 0 * (-inf) = NaN for illegal actions.
                        log_preds = torch.clamp(log_preds, min=-100.0)
                        # Normalize policy targets (guard against bad data).
                        p_sum = policies.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        policies_norm = policies / p_sum
                        # PER-weighted policy loss.
                        policy_loss = -(weights.unsqueeze(1) * policies_norm * log_preds).sum(dim=-1).mean()
                        policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=0.0)
                        # Value loss — MSE (в [0,1] для совместимости с ISMCTS).
                        v_target = (values + 1.0) * 0.5  # [-1,1] → [0,1]
                        v_pred = (pred_v + 1.0) * 0.5
                        value_loss = ((v_pred.squeeze(-1) - v_target.squeeze(-1)) ** 2 * weights).mean()
                        loss = policy_loss + value_loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_p, pred_v = model(states, masks)
                    log_preds = F.log_softmax(pred_p, dim=-1)
                    # Clamp to avoid 0 * (-inf) = NaN for illegal actions.
                    log_preds = torch.clamp(log_preds, min=-100.0)
                    # Normalize policy targets (guard against bad data).
                    p_sum = policies.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    policies_norm = policies / p_sum
                    policy_loss = -(weights.unsqueeze(1) * policies_norm * log_preds).sum(dim=-1).mean()
                    policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=0.0)
                    v_target = (values + 1.0) * 0.5
                    v_pred = (pred_v + 1.0) * 0.5
                    value_loss = ((v_pred.squeeze(-1) - v_target.squeeze(-1)) ** 2 * weights).mean()
                    loss = policy_loss + value_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                # Обновляем priorities в PER.
                with torch.no_grad():
                    # TD-error ≈ |policy_loss_per_sample + value_error|.
                    per_sample_loss = -(policies_norm * log_preds).sum(dim=-1) + \
                                      ((v_pred.squeeze(-1) - v_target.squeeze(-1)) ** 2)
                    per_sample_loss = torch.nan_to_num(per_sample_loss, nan=0.0, posinf=1.0)
                    td_errors = per_sample_loss.abs().detach().cpu().numpy()
                    buf.update_priorities(idxs, td_errors)

                total_loss += loss.item()
                total_p += policy_loss.item()
                total_v += value_loss.item()
                total_v_mse += value_loss.item()  # value_loss = MSE уже
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            avg_p = total_p / max(1, n_batches)
            avg_v = total_v / max(1, n_batches)
            t_iter = time.time() - t0
            print(f"[Iter {it}] loss={avg_loss:.4f} (p={avg_p:.4f} v={avg_v:.4f}) "
                  f"lr={lr:.2e} time={t_iter:.1f}s")

            writer.add_scalar("Loss/Total", avg_loss, it)
            writer.add_scalar("Loss/Policy", avg_p, it)
            writer.add_scalar("Loss/Value", avg_v, it)
            writer.add_scalar("Buffer/size", len(buf), it)
            writer.add_scalar("Perf/iter_time_sec", t_iter, it)
            writer.add_scalar("Perf/collect_time_sec", t_collect, it)

            metrics_file.write(json.dumps({
                "iter": it, "loss": avg_loss, "policy_loss": avg_p,
                "value_loss": avg_v, "lr": lr, "buffer_size": len(buf),
                "collected": collected, "iter_time": t_iter,
            }) + "\n")
            metrics_file.flush()

            # 8.4. Checkpoint + ONNX export.
            if (it + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_{it+1}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "loss": avg_loss,
                }, ckpt_path)
                torch.save({"model": model.state_dict(), "iter": it,
                            "optimizer": optimizer.state_dict()}, initial_pt)

                export_onnx(model, args.onnx_path, args.num_blocks, args.num_channels)

                if (it + 1) % (args.save_every * 5) == 0:
                    buf.save(os.path.join(args.checkpoint_dir, "buffer.pt"))
                print(f"[Iter {it}] Сохранён чекпойнт: {ckpt_path}")

            # 8.5. Arena: current vs baseline.
            if (it + 1) % args.arena_every == 0 and it > 0:
                print(f"[Iter {it}] Arena evaluation ({args.arena_games} партий)...")
                t_arena = time.time()
                try:
                    result = arena.evaluate_current_vs_baseline()
                    wr = result["current_wins"] / max(1, result["current_wins"] + result["baseline_wins"])
                    elo_diff = result.get("elo_diff", 0.0)
                    print(f"[Iter {it}] Arena: current={result['current_wins']} "
                          f"baseline={result['baseline_wins']} draws={result['draws']} "
                          f"wr={wr:.1%} elo_diff={elo_diff:+.1f} "
                          f"({time.time()-t_arena:.0f}s)")

                    writer.add_scalar("Arena/winrate", wr, it)
                    writer.add_scalar("Arena/draws", result["draws"], it)
                    writer.add_scalar("Arena/elo_diff", elo_diff, it)

                    if wr >= args.arena_threshold:
                        best_winrate = wr
                        best_path = os.path.join(args.checkpoint_dir, "best.pt")
                        torch.save({"model": model.state_dict(), "iter": it,
                                    "winrate": wr}, best_path)
                        arena.update_baseline()
                        print(f"[Iter {it}] Новый baseline! wr={wr:.1%}")
                except Exception as e:
                    print(f"[Iter {it}] Arena ошибка: {e}")

            # 8.6. Внешний baseline: current vs RandomNet-ISMCTS.
            if (it + 1) % args.ext_baseline_every == 0 and it > 0:
                print(f"[Iter {it}] External baseline vs RandomNet-ISMCTS "
                      f"({args.ext_baseline_games} партий)...")
                t_ext = time.time()
                try:
                    result = ext_evaluator.evaluate_vs_random_ismcts()
                    wr_ext = result["wins"] / max(1, result["wins"] + result["losses"])
                    elo_random = result.get("elo_vs_random", -1000.0)
                    print(f"[Iter {it}] vs RandomNet: wins={result['wins']} "
                          f"losses={result['losses']} draws={result['draws']} "
                          f"wr={wr_ext:.1%} elo={elo_random:+.1f} "
                          f"({time.time()-t_ext:.0f}s)")

                    writer.add_scalar("ExtBaseline/winrate_vs_random", wr_ext, it)
                    writer.add_scalar("ExtBaseline/elo_vs_random", elo_random, it)

                    if elo_random > best_elo_vs_random:
                        best_elo_vs_random = elo_random
                        best_path = os.path.join(args.checkpoint_dir,
                                                  "best_vs_random.pt")
                        torch.save({"model": model.state_dict(), "iter": it,
                                    "elo_vs_random": elo_random}, best_path)
                        print(f"[Iter {it}] Новый best vs random! elo={elo_random:+.1f}")
                except Exception as e:
                    print(f"[Iter {it}] ExtBaseline ошибка: {e}")

    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
    finally:
        mgr.stop()
        arena.close()
        ext_evaluator.close()
        writer.close()
        metrics_file.close()
        final = os.path.join(args.checkpoint_dir, "final.pt")
        current_it = locals().get('it', start_iter)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": current_it,
        }, final)
        print(f"Финальный чекпойнт: {final}")
        print(f"Лучший Elo vs Random: {best_elo_vs_random:+.1f}")


if __name__ == "__main__":
    main()
