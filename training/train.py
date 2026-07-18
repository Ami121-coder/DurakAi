"""
train.py — FIX #1 + FIX #6: авто-экспорт ONNX + включённая arena

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
  1. После каждого checkpoint-save АВТОМАТИЧЕСКИ экспортирует ONNX.
     Это замыкает цикл AlphaZero: воркеры подхватывают новую модель.
  2. Arena evaluation ВКЛЮЧЕНА — каждые N итераций сравниваем с baseline.
  3. Убран nan_to_num — используется finfo.min в модели (FIX #5).
  4. Добавлен параметр --onnx_path (куда экспортировать).
  5. Логирование winrate в TensorBoard.

Запуск:
  python train.py --iterations 1000 --batch_size 4096 --num_workers 6 --fp16
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import math
import argparse
import random
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
from arena import ArenaEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero-обучение для Durak")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs_per_iter", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--time_budget", type=float, default=0.5)
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--games_per_iter", type=int, default=4)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--onnx_path", type=str, default="checkpoints/current.onnx",
                   help="Куда экспортировать ONNX для self-play воркеров")
    p.add_argument("--log_dir", type=str, default="runs/durak_az")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arena_every", type=int, default=20)
    p.add_argument("--arena_games", type=int, default=40,
                   help="Партий в arena evaluation")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def export_onnx(model, output_path: str):
    """FIX #1: авто-экспорт ONNX после каждого checkpoint."""
    import export_onnx
    try:
        # Сохраняем временный .pt
        tmp_pt = os.path.join(os.path.dirname(output_path), "_tmp_export.pt")
        torch.save({"model": model.state_dict()}, tmp_pt)
        export_onnx.export(tmp_pt, output_path, num_blocks=8, num_channels=256)
        os.remove(tmp_pt)
        print(f"  [ONNX] Экспортирован: {output_path}")
        return True
    except Exception as e:
        print(f"  [ONNX] Ошибка экспорта: {e}")
        return False


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Устройство: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Модель
    model = build_model(num_blocks=8, num_channels=256).to(device)
    print(f"Параметров модели: {model.count_parameters():,}")

    # 2. Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda') if args.fp16 else None

    def cosine_with_warmup(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        progress = (it - args.warmup_iters) / max(1, args.iterations - args.warmup_iters)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))

    # 3. Replay buffer
    buf = ReplayBuffer(capacity=args.buffer_size)

    # 4. Resume
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        if os.path.exists(os.path.join(args.checkpoint_dir, "buffer.pt")):
            buf.load(os.path.join(args.checkpoint_dir, "buffer.pt"))
        print(f"Resumed с итерации {start_iter}, буфер {len(buf):,} позиций")

    # 5. TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # 6. FIX #1: экспортируем ONNX ДО запуска воркеров
    initial_pt = os.path.join(args.checkpoint_dir, "current.pt")
    if not os.path.exists(initial_pt):
        torch.save({"model": model.state_dict(), "iter": -1,
                    "optimizer": optimizer.state_dict()}, initial_pt)
    print("Экспорт ONNX при запуске...")
    export_onnx(model, args.onnx_path)

    # 7. Запускаем self-play воркеры (ВСЕГДА с OnnxNet в C++ ISMCTS)
    mgr = SelfPlayManager(
        num_workers=args.num_workers,
        onnx_path=args.onnx_path,
        games_per_iteration=args.games_per_iter,
        time_budget=args.time_budget,
        num_threads=args.num_threads,
    )
    mgr.start()

    # 8. Arena evaluator
    arena = ArenaEvaluator(
        onnx_path=args.onnx_path,
        time_budget=args.time_budget,
        num_games=args.arena_games,
    )

    best_winrate = 0.0

    try:
        for it in range(start_iter, args.iterations):
            t0 = time.time()

            # 8.1. Соберём примеры
            states_np, policies_np, values_np, masks_np = mgr.collect(
                max_examples=args.batch_size * 4, timeout=60.0)
            collected = len(states_np)

            dead = sum(1 for w in mgr.workers if not w.is_alive())
            if dead > 0:
                print(f"[ALARM] {dead} воркеров мертвы!")

            if collected > 0:
                buf.add(states_np, policies_np, values_np, masks_np)
            t_collect = time.time() - t0
            print(f"[Iter {it}] Собрано {collected} примеров за {t_collect:.1f}s, буфер={len(buf):,}")

            if len(buf) < args.batch_size:
                print(f"[Iter {it}] Буфер слишком мал — пропускаем обучение.")
                continue

            # 8.2. LR
            lr = cosine_with_warmup(it)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            writer.add_scalar("Train/learning_rate", lr, it)

            # 8.3. Обучение
            model.train()
            total_loss = total_p = total_v = 0.0
            n_batches = 0
            for epoch in range(args.epochs_per_iter):
                s_np, p_np, v_np, m_np = buf.sample(args.batch_size)
                states = torch.from_numpy(s_np).to(device)
                policies = torch.from_numpy(p_np).to(device)
                values = torch.from_numpy(v_np).to(device)
                masks = torch.from_numpy(m_np).to(device)

                optimizer.zero_grad()
                if args.fp16:
                    with torch.amp.autocast('cuda'):
                        pred_p, pred_v = model(states, masks)
                        log_preds = F.log_softmax(pred_p, dim=-1)
                        policy_loss = -(policies * log_preds).sum(dim=-1).mean()
                        value_loss = F.mse_loss(pred_v.squeeze(-1), values.squeeze(-1))
                        loss = policy_loss + value_loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_p, pred_v = model(states, masks)
                    log_preds = F.log_softmax(pred_p, dim=-1)
                    policy_loss = -(policies * log_preds).sum(dim=-1).mean()
                    value_loss = F.mse_loss(pred_v.squeeze(-1), values.squeeze(-1))
                    loss = policy_loss + value_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                total_loss += loss.item()
                total_p += policy_loss.item()
                total_v += value_loss.item()
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

            # 8.4. Checkpoint + ONNX export
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

                # FIX #1: экспортируем ONNX — воркеры подхватят новую модель
                export_onnx(model, args.onnx_path)

                if (it + 1) % (args.save_every * 5) == 0:
                    buf.save(os.path.join(args.checkpoint_dir, "buffer.pt"))
                print(f"[Iter {it}] Сохранён чекпойнт: {ckpt_path}")

            # 8.5. Arena evaluation
            if (it + 1) % args.arena_every == 0 and it > 0:
                print(f"[Iter {it}] Arena evaluation ({args.arena_games} партий)...")
                result = arena.evaluate_current_vs_baseline()
                wr = result["current_wins"] / max(1, result["current_wins"] + result["baseline_wins"])
                print(f"[Iter {it}] Arena: current={result['current_wins']} "
                      f"baseline={result['baseline_wins']} draws={result['draws']} "
                      f"winrate={wr:.1%}")
                writer.add_scalar("Arena/winrate", wr, it)
                writer.add_scalar("Arena/draws", result["draws"], it)

                if wr > best_winrate:
                    best_winrate = wr
                    best_path = os.path.join(args.checkpoint_dir, "best.pt")
                    torch.save({"model": model.state_dict(), "iter": it,
                                "winrate": wr}, best_path)
                    arena.update_baseline()
                    print(f"[Iter {it}] Новый baseline! winrate={wr:.1%}")

    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
    finally:
        mgr.stop()
        writer.close()
        final = os.path.join(args.checkpoint_dir, "final.pt")
        current_it = locals().get('it', start_iter)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": current_it,
        }, final)
        print(f"Финальный чекпойнт: {final}")


if __name__ == "__main__":
    main()
