import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from robustbench.utils import load_model


def set_seed(seed: int) -> None:
    """Cố định hạt giống ngẫu nhiên cho Python, NumPy và PyTorch.

    Parameters:
        seed (int): Giá trị seed cần cố định.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Đọc file cấu hình YAML thành dictionary.

    Parameters:
        config_path (str): Đường dẫn tới file .yaml.

    Returns:
        Dict[str, Any]: Dictionary chứa toàn bộ siêu tham số thực nghiệm.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_source_model(
    model_name: str,
    dataset: str,
    threat_model: str = "corruptions",
    model_dir: str = "./models",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Tải mô hình pre-trained nguồn từ RobustBench.

    Parameters:
        model_name (str): Tên mô hình trên RobustBench (VD: 'Standard').
        dataset (str): Tên dataset nguồn ('cifar10' hoặc 'cifar100').
        threat_model (str): Chuẩn model ('corruptions' hoặc 'Linf').
        model_dir (str): Thư mục lưu checkpoint tải về.
        device (Optional[torch.device]): Thiết bị đích (CPU/CUDA).

    Returns:
        nn.Module: Mô hình PyTorch đã nạp trọng số nguồn.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(
            model_name=model_name,
            dataset=dataset,
            threat_model=threat_model,
            model_dir=model_dir,
        )
    except KeyError:
        model = load_model(
            model_name=model_name,
            dataset=dataset,
            threat_model="Linf",
            model_dir=model_dir,
        )
    return model.to(device)


def init_wandb(
    config: Dict[str, Any],
    run_name: str,
    tags: Optional[List[str]] = None,
) -> Any:
    """Khởi tạo một lượt chạy mới trên dự án W&B công khai chung của nhóm.

    Parameters:
        config (Dict[str, Any]): Dictionary cấu hình cần lưu lên W&B.
        run_name (str): Tên hiển thị của lượt chạy trên dashboard W&B.
        tags (Optional[List[str]]): Các thẻ phân loại lượt chạy.

    Returns:
        wandb.sdk.wandb_run.Run: Đối tượng W&B run đang hoạt động.
    """
    return wandb.init(
        entity=config.get("wandb_entity", "huyle1156-hcmus"),
        project=config.get("wandb_project", "cotta-reproduction"),
        name=run_name,
        config=config,
        tags=tags or [],
    )


def log_corruption_metrics(
    step_idx: int,
    corruption: str,
    error_rate: float,
    extra: Optional[Dict[str, float]] = None,
) -> None:
    """Ghi nhận chỉ số lỗi của từng corruption lên W&B theo tiến trình thời gian t.

    Parameters:
        step_idx (int): Thứ tự bước corruption (1..15 hoặc 1..135).
        corruption (str): Tên loại corruption đang đánh giá.
        error_rate (float): Tỷ lệ lỗi phân loại (%) trên corruption hiện tại.
        extra (Optional[Dict[str, float]]): Các chỉ số phụ khác cần log thêm.

    Returns:
        None
    """
    payload: Dict[str, Any] = {
        "corruption_step": step_idx,
        "error_rate": error_rate,
        f"err/{corruption}": error_rate,
    }
    if extra:
        payload.update(extra)
    wandb.log(payload)