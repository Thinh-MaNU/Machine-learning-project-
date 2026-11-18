"""Kịch bản tái lập thực nghiệm Bảng 2 (Standard CIFAR10-to-CIFAR10C & Ablation Study).

Thực hiện:
1. Chạy đối chiếu 3 biến thể cắt bỏ (Ablation) của CoTTA trên 15 miền nhiễu Severity 5:
   - Variant 1: Weight-averaged Teacher only (use_aug=False, use_restore=False) -> Mục tiêu: 18.3%
   - Variant 2: Weight + Augmentation-averaged (use_aug=True, use_restore=False) -> Mục tiêu: 17.4%
   - Variant 3: Full CoTTA (use_aug=True, use_restore=True) -> Mục tiêu: 16.2% (±0.1%)
2. Tính độ lệch tương đối (Relative Deviation %) so với số liệu công bố trong Table 2.
3. Ghi nhận toàn bộ lượt chạy lên dự án W&B công khai của nhóm.
"""

import argparse
import copy
import inspect
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import src.datasets as ds_module
import src.utils as utils_module
from src.augmentations import build_cotta_augmenter
from src.cotta import build_cotta_from_config

# Số liệu chuẩn công bố tại Table 2 (trang 7) của bài báo CoTTA (CVPR 2022)
PAPER_TABLE2_BENCHMARKS: Dict[str, float] = {
    "weight_avg_only": 18.3,
    "weight_and_aug": 17.4,
    "full_cotta": 16.2,
}

CORRUPTION_ORDER: List[str] = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def load_domain_loaders_compat(cfg: Dict[str, object]) -> List[Tuple[str, DataLoader]]:
    """Tải dữ liệu từ Hugging Face và khởi tạo danh sách 15 DataLoader từ src/datasets.py."""
    data_root = str(cfg.get("data_dir", "./data/cifar10c"))

    if hasattr(ds_module, "prepare_dataset_from_hf"):
        prep_sig = inspect.signature(ds_module.prepare_dataset_from_hf)
        prep_pool = {
            "cfg": cfg,
            "config": cfg,
            "hf_repo_id": cfg.get("hf_repo_id", "YashP54/cifar10-c"),
            "repo_id": cfg.get("hf_repo_id", "YashP54/cifar10-c"),
            "data_dir": data_root,
            "data_root": data_root,
            "local_dir": data_root,
            "save_dir": data_root,
            "cache_dir": data_root,
            "dataset": cfg.get("dataset", "cifar10"),
            "dataset_name": cfg.get("dataset", "cifar10"),
        }
        prep_kwargs = {k: prep_pool[k]
                       for k in prep_sig.parameters if k in prep_pool}
        returned_path = ds_module.prepare_dataset_from_hf(**prep_kwargs)
        if returned_path is not None:
            data_root = str(returned_path)

    corruptions = getattr(ds_module, "CORRUPTIONS_ORDER", CORRUPTION_ORDER)
    loader_sig = inspect.signature(ds_module.load_corruption_dataloader)
    domain_loaders: List[Tuple[str, DataLoader]] = []

    for corr_name in corruptions:
        arg_pool = {
            "cfg": cfg,
            "config": cfg,
            "data_root": data_root,
            "data_dir": data_root,
            "root": data_root,
            "corruption": corr_name,
            "corruption_name": corr_name,
            "domain": corr_name,
            "name": corr_name,
            "severity": int(cfg.get("severity", 5)),
            "batch_size": int(cfg.get("batch_size", 200)),
            "shuffle": False,
            "dataset": cfg.get("dataset", "cifar10"),
        }
        kwargs = {k: arg_pool[k]
                  for k in loader_sig.parameters if k in arg_pool}
        loader = ds_module.load_corruption_dataloader(**kwargs)
        domain_loaders.append((corr_name, loader))

    return domain_loaders


def run_single_ablation_experiment(
    base_model: torch.nn.Module,
    domain_loaders: List[Tuple[str, DataLoader]],
    cfg: Dict[str, object],
    device: torch.device,
    variant_name: str,
    use_aug: bool,
    use_restore: bool,
    seed: int,
    loss_type: str = "ce",
    use_wandb: bool = True,
) -> Dict[str, float]:
    """Chạy 1 lượt tái lập CoTTA qua toàn bộ 15 miền nhiễu liên tiếp và log W&B.

    Args:
        base_model (torch.nn.Module): Mô hình nguồn WideResNet-28-10 tiền huấn luyện.
        domain_loaders (List[Tuple[str, DataLoader]]): Chuỗi 15 DataLoader theo thứ tự bài báo.
        cfg (Dict[str, object]): Cấu hình siêu tham số từ configs/default.yaml.
        device (torch.device): Thiết bị tính toán (CUDA GPU / CPU).
        variant_name (str): Tên biến thể ("weight_avg_only", "weight_and_aug", "full_cotta").
        use_aug (bool): Bật/tắt Augmentation-Averaged Pseudo-Labels.
        use_restore (bool): Bật/tắt Stochastic Restoration.
        seed (int): Hạt giống ngẫu nhiên của lượt chạy.
        loss_type (str): "ce" hoặc "symmetric_ce".
        use_wandb (bool): Bật/tắt ghi log lên Weights & Biases.

    Returns:
        Dict[str, float]: Từ điển Error Rate (%) từng miền, trung bình và độ lệch tương đối.
    """
    if hasattr(utils_module, "set_seed"):
        utils_module.set_seed(seed)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)

    augment_fn = build_cotta_augmenter(image_size=32) if use_aug else None
    adapter = build_cotta_from_config(
        model=copy.deepcopy(base_model),
        cfg=cfg,
        augment_fn=augment_fn,
        use_aug=use_aug,
        use_restore=use_restore,
        restoration_mode="random",
        loss_type=loss_type,
    ).to(device)

    run_name = f"repro_cifar10c_{variant_name}_{loss_type}_seed{seed}"
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            entity=str(cfg.get("wandb_entity", "huyle1156-hcmus")),
            project=str(cfg.get("wandb_project", "cotta-reproduction")),
            name=run_name,
            group=f"table2_{variant_name}",
            config={
                **cfg,
                "variant_name": variant_name,
                "use_aug": use_aug,
                "use_restore": use_restore,
                "loss_type": loss_type,
                "seed": seed,
            },
            reinit=True,
        )

    results: Dict[str, float] = {}
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

        err_rate = 100.0 * incorrect / max(1, total)
        results[domain_name] = round(err_rate, 2)
        print(
            f"  [{variant_name} | seed={seed}] ({step_idx + 1:02d}/15) {domain_name:18s}: {err_rate:.2f}%")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "corruption_idx": step_idx + 1,
                "corruption_name": domain_name,
                "error_rate": err_rate,
                "source_confidence": adapter.last_step_stats.get("source_confidence", 0.0),
                "aug_triggered": adapter.last_step_stats.get("aug_triggered", 0.0),
            })

    mean_err = float(np.mean(list(results.values())))
    paper_err = PAPER_TABLE2_BENCHMARKS[variant_name]
    rel_dev = 100.0 * (mean_err - paper_err) / paper_err

    results["mean_error"] = round(mean_err, 2)
    results["paper_error"] = paper_err
    results["relative_deviation_pct"] = round(rel_dev, 2)

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "mean_error_rate": mean_err,
            "paper_benchmark_error": paper_err,
            "relative_deviation_pct": rel_dev,
        })
        wandb.finish()

    return results


def load_source_model_compat(cfg: Dict[str, object], device: torch.device) -> torch.nn.Module:
    """Tự động khớp tham số và gọi hàm load_source_model() từ src/utils.py của nhóm.

    Args:
        cfg (Dict[str, object]): Từ điển cấu hình đọc từ file YAML.
        device (torch.device): Thiết bị thực thi (cuda hoặc cpu).

    Returns:
        torch.nn.Module: Mô hình nguồn tiền huấn luyện đã chuyển sang thiết bị tính toán.
    """
    sig = inspect.signature(utils_module.load_source_model)
    arg_pool = {
        "cfg": cfg,
        "config": cfg,
        "dataset": cfg.get("dataset", "cifar10"),
        "model_name": cfg.get("model_name", "Standard"),
        "model_dir": cfg.get("model_dir", "./models"),
        "threat_model": cfg.get("threat_model", "corruptions"),
        "device": device,
    }
    kwargs = {
        param_name: arg_pool[param_name]
        for param_name in sig.parameters
        if param_name in arg_pool
    }
    return utils_module.load_source_model(**kwargs).to(device)


def main() -> None:
    """Điều phối chạy tái lập Bảng 2 và lưu báo cáo số liệu ra thư mục outputs/."""
    parser = argparse.ArgumentParser(
        description="Tái lập thực nghiệm Bảng 2 CIFAR-10-C (CoTTA)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Đường dẫn file YAML")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44], help="Seeds cho Full CoTTA")
    parser.add_argument("--loss-type", type=str, default="ce",
                        choices=["ce", "symmetric_ce"])
    parser.add_argument("--no-wandb", action="store_true",
                        help="Tắt W&B nếu chỉ chạy thử cục bộ")
    args = parser.parse_args()

    cfg = utils_module.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị thực thi: {device}")

    base_model = load_source_model_compat(cfg, device)
    domain_loaders = load_domain_loaders_compat(cfg)

    variants = [
        ("weight_avg_only", False, False, [args.seeds[0]]),
        ("weight_and_aug", True, False, [args.seeds[0]]),
        ("full_cotta", True, True, args.seeds),
    ]

    all_summaries: Dict[str, object] = {}
    for variant_name, use_aug, use_restore, seed_list in variants:
        print(
            f"\n>>> Đang chạy biến thể: {variant_name} (seeds={seed_list}) <<<")
        seed_runs = []
        for s in seed_list:
            res = run_single_ablation_experiment(
                base_model=base_model,
                domain_loaders=domain_loaders,
                cfg=cfg,
                device=device,
                variant_name=variant_name,
                use_aug=use_aug,
                use_restore=use_restore,
                seed=s,
                loss_type=args.loss_type,
                use_wandb=not args.no_wandb,
            )
            seed_runs.append(res)

        mean_across_seeds = float(
            np.mean([r["mean_error"] for r in seed_runs]))
        std_across_seeds = float(np.std([r["mean_error"] for r in seed_runs]))
        paper_val = PAPER_TABLE2_BENCHMARKS[variant_name]
        rel_dev = 100.0 * (mean_across_seeds - paper_val) / paper_val

        all_summaries[variant_name] = {
            "paper_mean_error": paper_val,
            "ours_mean_error": round(mean_across_seeds, 2),
            "ours_std_error": round(std_across_seeds, 2),
            "relative_deviation_pct": round(rel_dev, 2),
            "runs": seed_runs,
        }

    Path("outputs").mkdir(exist_ok=True)
    out_file = Path("outputs") / f"reproduce_cifar10c_{args.loss_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== BẢNG TỔNG HỢP KẾT QUẢ TÁI LẬP (TABLE 2 — CIFAR-10-C) ===")
    print("| Biến thể CoTTA | Paper (%) | Ours (%) | Relative Deviation (%) |")
    print("| :--- | :---: | :---: | :---: |")
    for name, info in all_summaries.items():
        ours_str = (
            f"{info['ours_mean_error']:.2f} ± {info['ours_std_error']:.2f}"
            if info["ours_std_error"] > 0
            else f"{info['ours_mean_error']:.2f}"
        )
        print(
            f"| {name} | {info['paper_mean_error']:.2f} | {ours_str} | {info['relative_deviation_pct']:+.2f}% |")


if __name__ == "__main__":
    main()
