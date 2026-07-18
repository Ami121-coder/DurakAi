"""
export_onnx.py — экспорт PyTorch DurakResNet в ONNX для использования в C++ боте.

Входы ONNX-модели:
  - "state"      : float32[batch, 220]
  - "legal_mask" : float32[batch, 38]
Выходы:
  - "policy"     : float32[batch, 38]  — логиты (masked_fill применён)
  - "value"      : float32[batch, 1]   — в [-1, 1]

Совместимость с C++ onnx_net.cpp: имена входов/выходов должны совпадать.

Запуск:
  python export_onnx.py --checkpoint checkpoints/model_100.pt --output model.onnx
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_resnet import build_model


def export(checkpoint_path: str, output_path: str,
           num_blocks: int = 8, num_channels: int = 256,
           opset_version: int = 17):
    device = torch.device("cpu")  # ONNX export лучше делать на CPU.

    # Загружаем модель.
    model = build_model(num_blocks=num_blocks, num_channels=num_channels).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Модель загружена: {checkpoint_path}")

    # Демонстрационные входы.
    state = torch.randn(1, 220, device=device)
    legal_mask = torch.ones(1, 38, device=device)

    # Экспорт.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (state, legal_mask),
        output_path,
        input_names=["state", "legal_mask"],
        output_names=["policy", "value"],
        dynamic_axes={
            "state":      {0: "batch"},
            "legal_mask": {0: "batch"},
            "policy":     {0: "batch"},
            "value":      {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX экспортирован: {output_path}")

    # Sanity check: перезагрузим через onnxruntime и сравним с PyTorch.
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        # Сделаем случайный батч.
        test_state = np.random.randn(4, 220).astype(np.float32)
        test_mask = (np.random.rand(4, 38) > 0.3).astype(np.float32)
        out = sess.run(["policy", "value"],
                       {"state": test_state, "legal_mask": test_mask})

        # Сравним с PyTorch.
        with torch.no_grad():
            p_torch, v_torch = model(torch.from_numpy(test_state),
                                      torch.from_numpy(test_mask))
        p_diff = np.abs(out[0] - p_torch.numpy()).max()
        v_diff = np.abs(out[1] - v_torch.numpy()).max()
        print(f"ONNX vs PyTorch diff: policy={p_diff:.6f}, value={v_diff:.6f}")
        if p_diff > 1e-4 or v_diff > 1e-4:
            print("ПРЕДУПРЕЖДЕНИЕ: расхождения > 1e-4 — проверьте экспорт!")
        else:
            print("OK — ONNX-модель корректна.")
    except ImportError:
        print("onnxruntime не установлен — пропускаем sanity check.")
    except Exception as e:
        print(f"Ошибка при sanity check: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Путь к .pt чекпойнту")
    p.add_argument("--output", required=True, help="Путь к .onnx файлу")
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--num_channels", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    export(args.checkpoint, args.output, args.num_blocks, args.num_channels, args.opset)


if __name__ == "__main__":
    main()
