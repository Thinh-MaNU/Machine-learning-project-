"""Kịch bản thực nghiệm Phản biện 1 (Critique 1): Phân tích Stochastic Restoration.

Thực hiện 2 thí nghiệm chứng minh điểm yếu lý thuyết của mặt nạ Bernoulli đồng nhất:
1. Khảo sát độ nhạy của xác suất khôi phục p qua nhiều hạt giống ngẫu nhiên (seeds).
2. So sánh hiệu năng giữa Random Restoration (CoTTA gốc) và Gradient-Guided Restoration.
"""

import argparse
import copy
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.cotta import CoTTA, configure_model_for_cotta


def set_random_seed(seed: int) -> None:
    """Cố định toàn bộ hạt giống ngẫu nhiên để đảm bảo tính tái lập (Repeatability).

    Args:
        seed (int): Giá trị hạt giống ngẫu nhiên cần cố định.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_default_augmenter() -> Callable[[torch.Tensor], torch.Tensor]:
    """Khởi tạo hàm biến đổi ảnh tăng cường ngẫu nhiên phục vụ nhãn giả Teacher.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Hàm nhận batch tensor (B, C, H, W)
            và trả về batch ảnh đã biến đổi ngẫu nhiên.
    """
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(32, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    return transform


def evaluate_continual_stream(
    base_model: nn.Module,
    domain_loaders: Dict[str, DataLoader],
    device: torch.device,
    restoration_p: float,
    restoration_mode: str,
    lr: float = 1e-3,
    ema_alpha: float = 0.999,
    confidence_threshold: float = 0.92,
    num_augmentations: int = 32,
    seed: int = 42,
    use_wandb: bool = False,
    project_name: str = "IntroML-CoTTA",
) -> Dict[str, float]:
    """Chạy đánh giá CoTTA xuyên suốt chuỗi các miền nhiễu liên tục và ghi nhận Error Rate.

    Args:
        base_model (nn.Module): Mô hình tiền huấn luyện gốc trên miền nguồn.
        domain_loaders (Dict[str, DataLoader]): Từ điển ánh xạ tên miền nhiễu -> DataLoader.
        device (torch.device): Thiết bị tính toán (CPU hoặc CUDA GPU).
        restoration_p (float): Xác suất khôi phục trọng số p.
        restoration_mode (str): Chế độ khôi phục ("random" hoặc "gradient_guided").
        lr (float): Tốc độ học của bộ tối ưu Adam (mặc định: 1e-3).
        ema_alpha (float): Hệ số EMA của Teacher (mặc định: 0.999).
        confidence_threshold (float): Ngưỡng tự tin p_th (mặc định: 0.92).
        num_augmentations (int): Số phép biến đổi tăng cường N (mặc định: 32).
        seed (int): Hạt giống ngẫu nhiên của lượt chạy.
        use_wandb (bool): Bật/tắt việc đẩy kết quả lên Weights & Biases.
        project_name (str): Tên dự án trên Weights & Biases.

    Returns:
        Dict[str, float]: Từ điển chứa Error Rate (%) của từng miền và trung bình toàn chuỗi.
    """
    set_random_seed(seed)

    model = configure_model_for_cotta(copy.deepcopy(base_model)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    augment_fn = build_default_augmenter()

    cotta_adapter = CoTTA(
        model=model,
        optimizer=optimizer,
        augment_fn=augment_fn,
        ema_alpha=ema_alpha,
        restoration_p=restoration_p,
        confidence_threshold=confidence_threshold,
        num_augmentations=num_augmentations,
        restoration_mode=restoration_mode,
    )

    run_name = f"critique1_{restoration_mode}_p{restoration_p}_seed{seed}"
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "experiment": "critique_1_restoration",
                "restoration_mode": restoration_mode,
                "restoration_p": restoration_p,
                "lr": lr,
                "ema_alpha": ema_alpha,
                "confidence_threshold": confidence_threshold,
                "num_augmentations": num_augmentations,
                "seed": seed,
            },
            reinit=True,
        )

    domain_errors: Dict[str, float] = {}
    for domain_idx, (domain_name, loader) in enumerate(domain_loaders.items()):
        total_samples = 0
        total_incorrect = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            probs = cotta_adapter.forward_and_adapt(images)
            predictions = probs.argmax(dim=1)

            total_incorrect += int((predictions != labels).sum().item())
            total_samples += int(labels.size(0))

        error_rate = 100.0 * total_incorrect / max(1, total_samples)
        domain_errors[domain_name] = error_rate

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "domain_step": domain_idx + 1,
                "domain_name": domain_name,
                "domain_error_rate": error_rate,
                "last_batch_loss": cotta_adapter.last_step_stats.get("loss", 0.0),
            })

    mean_error = float(np.mean(list(domain_errors.values())))
    domain_errors["mean_error"] = mean_error

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"mean_continual_error_rate": mean_error})
        wandb.finish()

    return domain_errors


def create_smoke_test_environment() -> Tuple[nn.Module, Dict[str, DataLoader]]:
    """Tạo mô hình nhẹ và dữ liệu giả lập để kiểm thử nhanh luồng code khi chưa tải CIFAR-10-C.

    Returns:
        Tuple[nn.Module, Dict[str, DataLoader]]: Mô hình CNN nhỏ và 3 miền dữ liệu giả lập.
    """
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10),
    )
    loaders: Dict[str, DataLoader] = {}
    for domain in ["gaussian_noise", "defocus_blur", "fog"]:
        x = torch.rand(64, 3, 32, 32)
        y = torch.randint(0, 10, (64,))
        loaders[domain] = DataLoader(TensorDataset(
            x, y), batch_size=32, shuffle=False)

    return dummy_model, loaders


def main() -> None:
    """Hàm điều phối chạy Thí nghiệm 1A (Sensitivity Sweep) và 1B (Guided vs Random)."""
    parser = argparse.ArgumentParser(
        description="Chạy thực nghiệm Critique 1 cho CoTTA")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Chạy kiểm thử nhanh bằng dữ liệu giả lập")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Đẩy log thực nghiệm lên W&B")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44], help="Danh sách random seeds")
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
        help="Danh sách xác suất khôi phục p cần khảo sát",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        print("[Smoke Test] Khởi tạo môi trường kiểm thử nhanh...")
        base_model, domain_loaders = create_smoke_test_environment()
        num_augs = 4
    else:
        # Khi Thành viên 2 hoàn thành src/datasets.py, tự động nạp CIFAR-10-C và RobustBench
        from robustbench.utils import load_model
        from src.datasets import get_cifar10c_continual_loaders

        base_model = load_model(model_name="Standard",
                                dataset="cifar10", threat_model="corruptions")
        domain_loaders = get_cifar10c_continual_loaders(
            batch_size=200, severity=5)
        num_augs = 32

    print("\n=== THÍ NGHIỆM 1A & 1B: SO SÁNH RANDOM vs. GRADIENT-GUIDED RESTORATION ===")
    for mode in ["random", "gradient_guided"]:
        for p in args.p_values:
            seed_errors: List[float] = []
            for seed in args.seeds:
                stats = evaluate_continual_stream(
                    base_model=base_model,
                    domain_loaders=domain_loaders,
                    device=device,
                    restoration_p=p,
                    restoration_mode=mode,
                    num_augmentations=num_augs,
                    seed=seed,
                    use_wandb=args.use_wandb,
                )
                seed_errors.append(stats["mean_error"])

            mean_err = float(np.mean(seed_errors))
            std_err = float(np.std(seed_errors))
            print(
                f"Mode: {mode:15s} | p = {p:<6.3f} | Mean Error: {mean_err:.2f}% ± {std_err:.2f}%")


if __name__ == "__main__":
    main()
