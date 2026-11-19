"""Kịch bản tái lập thực nghiệm Bảng 3 (Gradually Changing CIFAR10-to-CIFAR10C).

Sử dụng hàm get_gradual_sequence() từ src/datasets.py để đưa mô hình đi qua
135 chặng tăng-giảm độ khó liên tục (1 -> 2 -> 3 -> 4 -> 5 -> 4 -> 3 -> 2 -> 1).
Mục tiêu tham chiếu Table 3: 10.4 ± 0.2%.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import src.datasets as ds_module
import src.utils as utils_module
from experiments.reproduce_cifar10c import load_source_model_compat
from src.augmentations import build_cotta_augmenter
from src.cotta import build_cotta_from_config

PAPER_TABLE3_COTTA = 10.4


def run_gradual_sequence(
    cfg: Dict[str, object],
    order_seed: int,
    model_seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, object]:
    """Chạy CoTTA trên một chuỗi 135 chặng Gradual CIFAR10-C.

    Args:
        cfg (Dict[str, object]): Cấu hình đọc từ configs/default.yaml.
        order_seed (int): Seed tạo thứ tự xáo trộn 15 loại corruption.
        model_seed (int): Seed cố định trọng số và quá trình ngẫu nhiên.
        use_wandb (bool): Bật/tắt ghi log lên W&B.

    Returns:
        Dict[str, object]: Error Rate trung bình qua 135 chặng và độ lệch tương đối.
    """
    utils_module.set_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = ds_module.prepare_dataset_from_hf(
        str(cfg.get("hf_repo_id", "YashP54/cifar10-c")),
        str(cfg.get("data_dir", "./data/cifar10c")),
    )
    base_model = load_source_model_compat(cfg, device)
    augment_fn = build_cotta_augmenter(image_size=32)
    adapter = build_cotta_from_config(
        model=copy.deepcopy(base_model),
        cfg=cfg,
        augment_fn=augment_fn,
        use_aug=True,
        use_restore=True,
        restoration_mode="random",
        loss_type="ce",
    ).to(device)

    sequence = ds_module.get_gradual_sequence(seed=order_seed)
    run_name = f"repro_gradual_c10c_order{order_seed}_seed{model_seed}"

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            entity=str(cfg.get("wandb_entity", "huyle1156-hcmus")),
            project=str(cfg.get("wandb_project", "cotta-reproduction")),
            name=run_name,
            tags=["cifar10c", "gradual", "table3"],
            config={**cfg, "order_seed": order_seed, "model_seed": model_seed},
            reinit=True,
        )

    stage_errors = []
    for stage_idx, (corr_name, severity) in enumerate(sequence, start=1):
        loader = ds_module.load_corruption_dataloader(
            data_root=data_root,
            corruption=corr_name,
            severity=severity,
            batch_size=int(cfg.get("batch_size", 200)),
            shuffle=False,
        )
        incorrect = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            probs = adapter.forward_and_adapt(images)
            preds = probs.argmax(dim=1)
            incorrect += int((preds != labels).sum().item())
            total += int(labels.size(0))

        err_rate = 100.0 * incorrect / max(1, total)
        stage_errors.append(err_rate)
        if stage_idx % 9 == 0 or severity == 5:
            print(f"  [Gradual | {stage_idx:03d}/135] {corr_name:18s} (sev={severity}): {err_rate:.2f}%")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "stage_idx": stage_idx,
                "severity": severity,
                "error_rate": err_rate,
                "aug_triggered": adapter.last_step_stats.get("aug_triggered", 0.0),
            })

        del loader

    mean_err = float(np.mean(stage_errors))
    rel_dev = 100.0 * (mean_err - PAPER_TABLE3_COTTA) / PAPER_TABLE3_COTTA

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "mean_error_rate": mean_err,
            "paper_benchmark_error": PAPER_TABLE3_COTTA,
            "relative_deviation_pct": rel_dev,
        })
        wandb.finish()

    return {
        "order_seed": order_seed,
        "paper_mean_error": PAPER_TABLE3_COTTA,
        "ours_mean_error": round(mean_err, 2),
        "relative_deviation_pct": round(rel_dev, 2),
        "stage_errors": [round(x, 2) for x in stage_errors],
    }


def main() -> None:
    """Điểm khởi chạy script tái lập Bảng 3 (Gradual CIFAR10-C)."""
    parser = argparse.ArgumentParser(description="Reproduce Table 3: Gradual CIFAR10-C")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = utils_module.load_config(args.config)
    summary = run_gradual_sequence(
        cfg=cfg,
        order_seed=args.order_seed,
        model_seed=args.model_seed,
        use_wandb=not args.no_wandb,
    )

    Path("outputs").mkdir(exist_ok=True)
    out_file = Path("outputs") / f"reproduce_gradual_order{args.order_seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== KẾT QUẢ TÁI LẬP TABLE 3 (GRADUAL CIFAR10-C) ===")
    print(f"Paper: {PAPER_TABLE3_COTTA:.2f}% | Ours: {summary['ours_mean_error']:.2f}% | "
          f"Relative Deviation: {summary['relative_deviation_pct']:+.2f}%")


if __name__ == "__main__":
    main()