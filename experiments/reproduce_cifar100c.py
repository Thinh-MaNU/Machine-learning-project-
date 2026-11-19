"""Kịch bản tái lập thực nghiệm Bảng 4 (Standard CIFAR100-to-CIFAR100C).

Đánh giá Full CoTTA trên kiến trúc ResNeXt-29 qua 15 miền nhiễu Severity 5.
Mục tiêu tham chiếu trong Table 4 (trang 8) bài báo CoTTA: 32.5%.
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

import src.utils as utils_module
from experiments.reproduce_cifar10c import (
    load_domain_loaders_compat,
    load_source_model_compat,
)
from src.augmentations import build_cotta_augmenter
from src.cotta import build_cotta_from_config

PAPER_TABLE4_COTTA = 32.5


def run_cifar100c(
    cfg: Dict[str, object],
    seed: int,
    loss_type: str = "ce",
    use_wandb: bool = True,
) -> Dict[str, object]:
    """Chạy tái lập Full CoTTA trên bộ dữ liệu CIFAR100-to-CIFAR100C.

    Args:
        cfg (Dict[str, object]): Từ điển cấu hình đọc từ configs/cifar100c.yaml.
        seed (int): Hạt giống ngẫu nhiên cho lượt chạy.
        loss_type (str): Hàm mất mát nhất quán ("ce" hoặc "symmetric_ce").
        use_wandb (bool): Bật/tắt ghi log lên Weights & Biases.

    Returns:
        Dict[str, object]: Kết quả Error Rate (%) từng miền, trung bình và độ lệch tương đối.
    """
    utils_module.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CIFAR100-C] Thiết bị thực thi: {device}")

    base_model = load_source_model_compat(cfg, device)
    domain_loaders = load_domain_loaders_compat(cfg)

    augment_fn = build_cotta_augmenter(image_size=32)
    adapter = build_cotta_from_config(
        model=copy.deepcopy(base_model),
        cfg=cfg,
        augment_fn=augment_fn,
        use_aug=True,
        use_restore=True,
        restoration_mode="random",
        loss_type=loss_type,
    ).to(device)

    run_name = f"repro_cifar100c_full_cotta_{loss_type}_seed{seed}"
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            entity=str(cfg.get("wandb_entity", "huyle1156-hcmus")),
            project=str(cfg.get("wandb_project", "cotta-reproduction")),
            name=run_name,
            tags=["cifar100c", "table4", "cotta-full"],
            config={**cfg, "loss_type": loss_type, "seed": seed},
            reinit=True,
        )

    results: Dict[str, float] = {}
    aug_rates = []
    for step_idx, (domain_name, loader) in enumerate(domain_loaders):
        incorrect = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            probs = adapter.forward_and_adapt(images)
            preds = probs.argmax(dim=1)
            incorrect += int((preds != labels).sum().item())
            total += int(labels.size(0))
            aug_rates.append(adapter.last_step_stats.get("aug_triggered", 0.0))

        err_rate = 100.0 * incorrect / max(1, total)
        results[domain_name] = round(err_rate, 2)
        print(f"  [CIFAR100-C | seed={seed}] ({step_idx + 1:02d}/15) {domain_name:18s}: {err_rate:.2f}%")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "corruption_idx": step_idx + 1,
                "corruption_name": domain_name,
                "error_rate": err_rate,
                "source_confidence": adapter.last_step_stats.get("source_confidence", 0.0),
                "aug_triggered": adapter.last_step_stats.get("aug_triggered", 0.0),
            })

    mean_err = float(np.mean(list(results.values())))
    rel_dev = 100.0 * (mean_err - PAPER_TABLE4_COTTA) / PAPER_TABLE4_COTTA
    avg_aug_rate = 100.0 * float(np.mean(aug_rates)) if aug_rates else 0.0

    summary: Dict[str, object] = {
        "paper_mean_error": PAPER_TABLE4_COTTA,
        "ours_mean_error": round(mean_err, 2),
        "relative_deviation_pct": round(rel_dev, 2),
        "avg_aug_trigger_pct": round(avg_aug_rate, 2),
        "per_corruption": results,
    }

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "mean_error_rate": mean_err,
            "paper_benchmark_error": PAPER_TABLE4_COTTA,
            "relative_deviation_pct": rel_dev,
            "avg_aug_trigger_pct": avg_aug_rate,
        })
        wandb.finish()

    return summary


def main() -> None:
    """Điểm khởi chạy script tái lập Bảng 4 (CIFAR100-to-CIFAR100C)."""
    parser = argparse.ArgumentParser(description="Reproduce Table 4: CIFAR100-to-CIFAR100C")
    parser.add_argument("--config", type=str, default="configs/cifar100c.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "symmetric_ce"])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = utils_module.load_config(args.config)
    summary = run_cifar100c(
        cfg=cfg,
        seed=args.seed,
        loss_type=args.loss_type,
        use_wandb=not args.no_wandb,
    )

    Path("outputs").mkdir(exist_ok=True)
    out_file = Path("outputs") / f"reproduce_cifar100c_{args.loss_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== KẾT QUẢ TÁI LẬP TABLE 4 (CIFAR100-C) ===")
    print(f"Paper: {PAPER_TABLE4_COTTA:.2f}% | Ours: {summary['ours_mean_error']:.2f}% | "
          f"Relative Deviation: {summary['relative_deviation_pct']:+.2f}%")


if __name__ == "__main__":
    main()