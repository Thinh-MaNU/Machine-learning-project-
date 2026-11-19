"""Kịch bản khảo sát độ nhạy siêu tham số (Sensitivity Sweep) cho Core Scope.

Thay thế cho các thực nghiệm nặng đã lược bỏ (ImageNet-C, Cityscapes-to-ACDC) bằng
phân tích độ nhạy của các tham số quan trọng: p_th, restore_p, alpha, batch_size.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

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


def run_sweep_point(
    base_model: torch.nn.Module,
    cfg: Dict[str, object],
    param_name: str,
    param_value: float,
    seed: int = 42,
    use_wandb: bool = True,
) -> Dict[str, object]:
    """Chạy Full CoTTA trên 15 miền CIFAR10-C tại một điểm cấu hình siêu tham số.

    Args:
        base_model (torch.nn.Module): Mô hình nguồn tiền huấn luyện.
        cfg (Dict[str, object]): Cấu hình nền đọc từ configs/default.yaml.
        param_name (str): Tên siêu tham số đang khảo sát ('p_th', 'restore_p', 'alpha', 'batch_size').
        param_value (float): Giá trị thử nghiệm của siêu tham số.
        seed (int): Seed cố định cho mọi điểm sweep.
        use_wandb (bool): Bật/tắt ghi log lên W&B.

    Returns:
        Dict[str, object]: Kết quả Error Rate trung bình và tỷ lệ kích hoạt augmentation.
    """
    utils_module.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cfg = copy.deepcopy(cfg)
    run_cfg[param_name] = int(param_value) if param_name == "batch_size" else float(param_value)

    domain_loaders = load_domain_loaders_compat(run_cfg)
    augment_fn = build_cotta_augmenter(image_size=32)
    adapter = build_cotta_from_config(
        model=copy.deepcopy(base_model),
        cfg=run_cfg,
        augment_fn=augment_fn,
        use_aug=True,
        use_restore=float(run_cfg.get("restore_p", 0.01)) > 0.0,
        restoration_mode="random",
        loss_type="ce",
    ).to(device)

    run_name = f"sweep_{param_name}_{param_value}_seed{seed}"
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            entity=str(run_cfg.get("wandb_entity", "huyle1156-hcmus")),
            project=str(run_cfg.get("wandb_project", "cotta-reproduction")),
            name=run_name,
            group=f"sweep_{param_name}",
            tags=["sensitivity-sweep", param_name],
            config={**run_cfg, "sweep_param": param_name, "sweep_value": param_value},
            reinit=True,
        )

    per_domain: Dict[str, float] = {}
    aug_flags: List[float] = []
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
            aug_flags.append(adapter.last_step_stats.get("aug_triggered", 0.0))

        err_rate = 100.0 * incorrect / max(1, total)
        per_domain[domain_name] = round(err_rate, 2)
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"corruption_idx": step_idx + 1, "error_rate": err_rate})

    mean_err = float(np.mean(list(per_domain.values())))
    aug_pct = 100.0 * float(np.mean(aug_flags)) if aug_flags else 0.0
    print(f"  [Sweep {param_name}={param_value}] Mean Error: {mean_err:.2f}% | Aug Trigger: {aug_pct:.1f}%")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "mean_error_rate": mean_err,
            "avg_aug_trigger_pct": aug_pct,
            "sweep_value": float(param_value),
        })
        wandb.finish()

    return {
        "param": param_name,
        "value": param_value,
        "mean_error": round(mean_err, 2),
        "avg_aug_trigger_pct": round(aug_pct, 2),
        "per_corruption": per_domain,
    }


def main() -> None:
    """Điều phối chạy Sensitivity Sweep và lưu kết quả ra thư mục outputs/."""
    parser = argparse.ArgumentParser(description="Sensitivity Sweep for CoTTA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--param", type=str, required=True, choices=["p_th", "restore_p", "alpha", "batch_size"])
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = utils_module.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_source_model_compat(cfg, device)

    sweep_results = []
    for val in args.values:
        print(f"\n>>> Đang khảo sát {args.param} = {val} <<<")
        res = run_sweep_point(
            base_model=base_model,
            cfg=cfg,
            param_name=args.param,
            param_value=val,
            seed=args.seed,
            use_wandb=not args.no_wandb,
        )
        sweep_results.append(res)

    Path("outputs").mkdir(exist_ok=True)
    out_file = Path("outputs") / f"sensitivity_{args.param}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== TỔNG HỢP SENSITIVITY SWEEP ({args.param}) ===")
    for item in sweep_results:
        print(f"  {args.param} = {item['value']} -> Mean Error: {item['mean_error']:.2f}% (Aug: {item['avg_aug_trigger_pct']:.1f}%)")


if __name__ == "__main__":
    main()