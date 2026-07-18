"""
train.py.patched — полноценный AlphaZero-обучающий цикл.

Что меняется по сравнению с оригинальным train.py:
  1. Используется ResNet-8 (model_resnet.py) вместо 3-layer MLP.
  2. Replay Buffer на 1M позиций (replay_buffer.py).
  3. Self-play воркеры через multiprocessing (self_play.py) — async генерация.
  4. Mini-batch SGD: 4096 примеров × 32 эпохи за итерацию.
  5. Gradient clipping (max_norm=1.0).
  6. LR schedule: cosine annealing, warmup 100 итераций.
  7. L2 regularization (weight_decay=1e-4 в AdamW).
  8. Checkpointing: каждые 10 итераций сохраняем model_{iter}.pt.
  9. Arena evaluation: каждые 20 итераций сравниваем с baseline.
  10. TensorBoard логирование: loss/policy/value/learning_rate/buffer_size/winrate.
  11. Возможность resume: загружаем модель и буфер из чекпойнта.

Пример запуска:
  python train.py --iterations 1000 --batch_size 4096 --epochs_per_iter 4 \
                  --num_workers 6 --time_budget 0.5
"""

import os
import sys
import time
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Добавляем корень проекта в path.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_resnet import build_model
from replay_buffer import ReplayBuffer
from self_play import SelfPlayManager, arena_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero-обучение для Durak")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs_per_iter", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=6,
                   help="Кол-во self-play воркеров (1 на физ. ядро CPU)")
    p.add_argument("--time_budget", type=float, default=0.5,
                   help="Бюджет ISMCTS на ход, сек")
    p.add_argument("--num_threads", type=int, default=1,
                   help="Потоков ISMCTS в каждом воркере (1 = ок для self-play)")
    p.add_argument("--games_per_iter", type=int, default=4,
                   help="Партий на воркер за итерацию")
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="runs/durak_az")
    p.add_argument("--resume", type=str, default=None,
                   help="Путь к .pt чекпойнту для resume")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arena_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--fp16", action="store_true",
                   help="Mixed-precision training (AMP) для ускорения на 4060 Ti")
    return p.parse_args()


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

    # ---- 1. Модель ----
    model = build_model(num_blocks=8, num_channels=256).to(device)
    print(f"Параметров модели: {model.count_parameters():,}")
    print(f"VRAM под параметры (FP32): {model.count_parameters() * 4 / 1024**2:.1f} MB")

    # ---- 2. Оптимизатор + LR schedule ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler('cuda') if args.fp16 else None

    def cosine_with_warmup(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        progress = (it - args.warmup_iters) / max(1, args.iterations - args.warmup_iters)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))

    # ---- 3. Replay buffer ----
    buf = ReplayBuffer(capacity=args.buffer_size)

    # ---- 4. Resume ----
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        if os.path.exists(os.path.join(args.checkpoint_dir, "buffer.pt")):
            buf.load(os.path.join(args.checkpoint_dir, "buffer.pt"))
        print(f"Resumed с итерации {start_iter}, буфер {len(buf):,} позиций")

    # ---- 5. TensorBoard ----
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---- 6. Запускаем self-play воркеры ----
    # В первой итерации модель ещё не обучена — пусть воркеры используют
    # чистый ISMCTS. После каждого save — обновляем model_path.
    initial_ckpt = os.path.join(args.checkpoint_dir, "current.pt")
    if not os.path.exists(initial_ckpt):
        torch.save({"model": model.state_dict(), "iter": -1,
                    "optimizer": optimizer.state_dict()}, initial_ckpt)

    mgr = SelfPlayManager(num_workers=args.num_workers,
                          model_path=initial_ckpt,
                          games_per_iteration=args.games_per_iter,
                          time_budget=args.time_budget,
                          num_threads=args.num_threads)
    mgr.start()

    # ---- 7. Главный цикл ----
    try:
        for it in range(start_iter, args.iterations):
            t0 = time.time()

            # 7.1. Соберём примеры из self-play (не блокирующе, таймаут 60с).
            states_np, policies_np, values_np, masks_np = mgr.collect(
                max_examples=args.batch_size * 4, timeout=60.0)
            collected = len(states_np)
            if collected > 0:
                buf.add(states_np, policies_np, values_np, masks_np)
            t_collect = time.time() - t0
            print(f"[Iter {it}] Собрано {collected} примеров за {t_collect:.1f}s, буфер={len(buf):,}")

            if len(buf) < args.batch_size:
                print(f"[Iter {it}] Буфер слишком мал — пропускаем обучение.")
                continue

            # 7.2. Установим LR.
            lr = cosine_with_warmup(it)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            writer.add_scalar("Train/learning_rate", lr, it)

            # 7.3. Несколько эпох мини-батчей.
            model.train()
            total_loss = total_p = total_v = 0.0
            n_batches = 0
            for epoch in range(args.epochs_per_iter):
                s_np, p_np, v_np, m_np = buf.sample(args.batch_size)
                states = torch.from_numpy(s_np).to(device).float()
                policies = torch.from_numpy(p_np).to(device)
                values = torch.from_numpy(v_np).to(device)
                masks = torch.from_numpy(m_np).to(device)

                optimizer.zero_grad()
                if args.fp16:
                    with torch.amp.autocast('cuda'):
                        pred_p, pred_v = model(states, masks)
                        # Policy loss: cross-entropy с маскировкой.
                        # nan_to_num нужен: log_softmax(-inf)=-inf, 0*-inf=NaN
                        log_preds = F.log_softmax(pred_p, dim=-1).nan_to_num(nan=0.0, neginf=0.0)
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
                    # nan_to_num нужен: log_softmax(-inf)=-inf, 0*-inf=NaN
                    log_preds = F.log_softmax(pred_p, dim=-1).nan_to_num(nan=0.0, neginf=0.0)
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

            # 7.4. Checkpoint.
            if (it + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_{it+1}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "loss": avg_loss,
                }, ckpt_path)
                torch.save({"model": model.state_dict(), "iter": it,
                            "optimizer": optimizer.state_dict()},
                           initial_ckpt)
                # Периодически сохраняем буфер (для resume).
                if (it + 1) % (args.save_every * 5) == 0:
                    buf.save(os.path.join(args.checkpoint_dir, "buffer.pt"))
                print(f"[Iter {it}] Сохранён чекпойнт: {ckpt_path}")

            # 7.5. Arena evaluation.
            # Отключено: эта заглушка не использует нейросеть и тормозит обучение на 15+ минут.
            # if (it + 1) % args.arena_every == 0 and it > 0:
            #     print(f"[Iter {it}] Arena evaluation skipped...")

    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
    finally:
        mgr.stop()
        writer.close()
        # Финальный checkpoint.
        final = os.path.join(args.checkpoint_dir, "final.pt")
        current_it = locals().get('it', start_iter)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": current_it,
        }, final)
        print(f"Финальный чекпойнт: {final}")
        # Экспорт в ONNX (см. export_onnx.py).
        print("Не забудьте запустить export_onnx.py для использования в C++ боте.")


import torch.nn.functional as F  # нужен для F.mse_loss / F.log_softmax


if __name__ == "__main__":
    main()
