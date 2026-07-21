"""
export_onnx.py — обновлён под усиленную SE-модель.

Изменения:
  1. Принимает num_blocks=10, num_channels=256 по умолчанию (как в train.py).
  2. Проверка model.eval() перед экспортом (важно для BatchNorm).
  3. Sanity check с uint8 input (как из C++ bindings).
  4. Dynamic batch axis.
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_resnet import build_model


def export(checkpoint_path: str, output_path: str,
           num_blocks: int = 10, num_channels: int = 256,
           opset_version: int = 17):
    device = torch.device("cpu")

    model = build_model(num_blocks=num_blocks, num_channels=num_channels).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Модель загружена: {checkpoint_path} "
          f"(blocks={num_blocks}, channels={num_channels})")

    # Демонстрационные входы — float32 как из C++ (нормализуем там).
    state = torch.rand(1, 220, dtype=torch.float32)
    legal_mask = torch.ones(1, 38, dtype=torch.float32)

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

    # Sanity check.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        # Тест с uint8-like (0..255 как в bindings.cpp, модель нормализует внутри).
        test_state = (np.random.rand(4, 220) * 255).astype(np.float32)
        test_mask = (np.random.rand(4, 38) > 0.3).astype(np.float32)
        out = sess.run(["policy", "value"],
                       {"state": test_state, "legal_mask": test_mask})

        with torch.no_grad():
            # Модель ожидает uint8 или float; даём float (как в onnx_net.cpp).
            p_torch, v_torch = model(torch.from_numpy(test_state),
                                      torch.from_numpy(test_mask))
        p_diff = np.abs(out[0] - p_torch.numpy()).max()
        v_diff = np.abs(out[1] - v_torch.numpy()).max()
        print(f"ONNX vs PyTorch diff: policy={p_diff:.6f}, value={v_diff:.6f}")
        if p_diff > 1e-3 or v_diff > 1e-3:
            print("ПРЕДУПРЕЖДЕНИЕ: расхождения > 1e-3 — проверьте экспорт!")
        else:
            print("OK — ONNX-модель корректна.")

        # Проверим, что нет NaN/Inf.
        assert np.isfinite(out[0]).all(), "NaN/Inf в policy!"
        assert np.isfinite(out[1]).all(), "NaN/Inf в value!"
        print("OK — нет NaN/Inf")

    except ImportError:
        print("onnxruntime не установлен — пропускаем sanity check.")
    except Exception as e:
        print(f"Ошибка при sanity check: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--num_channels", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    export(args.checkpoint, args.output, args.num_blocks, args.num_channels, args.opset)


if __name__ == "__main__":
    main()
