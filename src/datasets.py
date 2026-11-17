import os
import random
import tarfile
from typing import List, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, TensorDataset

# Thứ tự 15 loại corruption chuẩn theo Bảng 2 và Bảng 4 của bài báo CoTTA
CORRUPTIONS_ORDER: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def prepare_dataset_from_hf(repo_id: str, local_dir: str) -> str:
    """Tải tự động dữ liệu CIFAR-C từ Hugging Face và giải nén nếu là file .tar.

    Parameters:
        repo_id (str): ID dataset công khai trên Hugging Face (VD: 'YashP54/cifar10-c').
        local_dir (str): Thư mục cục bộ lưu trữ dữ liệu sau khi tải.

    Returns:
        str: Đường dẫn thư mục thực tế chứa các file .npy (gồm labels.npy và 15 corruptions).
    """
    os.makedirs(local_dir, exist_ok=True)

    for root, _, files in os.walk(local_dir):
        if "labels.npy" in files:
            return root

    print(f"[Datasets] Đang tải tự động từ Hugging Face: {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.npy", "**/*.npy", "*.tar", "*.tar.gz", "*.tgz"],
    )

    for root, _, files in os.walk(local_dir):
        for fname in files:
            if fname.endswith((".tar", ".tar.gz", ".tgz")):
                tar_path = os.path.join(root, fname)
                print(f"[Datasets] Đang giải nén {tar_path}...")
                with tarfile.open(tar_path, "r:*") as tar:
                    tar.extractall(path=local_dir)

    for root, _, files in os.walk(local_dir):
        if "labels.npy" in files:
            return root

    raise FileNotFoundError(
        f"Không tìm thấy labels.npy trong {local_dir} sau khi tải từ {repo_id}."
    )


def load_corruption_arrays(
    data_root: str, corruption: str, severity: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Đọc dữ liệu thô (.npy) của một loại corruption ở mức severity xác định.

    Parameters:
        data_root (str): Thư mục chứa các file .npy.
        corruption (str): Tên loại corruption (thuộc CORRUPTIONS_ORDER).
        severity (int): Mức độ nhiễu từ 1 đến 5. Mặc định là 5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Bộ (images, labels) với ảnh chuẩn hóa [0.0, 1.0].
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity phải nằm trong khoảng [1, 5], nhận được: {severity}")

    img_path = os.path.join(data_root, f"{corruption}.npy")
    lbl_path = os.path.join(data_root, "labels.npy")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Thiếu file corruption: {img_path}")
    if not os.path.exists(lbl_path):
        raise FileNotFoundError(f"Thiếu file nhãn: {lbl_path}")

    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000

    images_np = np.load(img_path)[start_idx:end_idx]
    labels_np = np.load(lbl_path)[start_idx:end_idx]

    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    labels_tensor = torch.from_numpy(labels_np).long()
    return images_tensor, labels_tensor


def load_corruption_dataloader(
    data_root: str,
    corruption: str,
    severity: int = 5,
    batch_size: int = 200,
    shuffle: bool = False,
) -> DataLoader:
    """Tạo PyTorch DataLoader cho một loại corruption và mức severity cụ thể.

    Parameters:
        data_root (str): Thư mục chứa các file .npy.
        corruption (str): Tên loại corruption.
        severity (int): Mức độ nhiễu từ 1 đến 5. Mặc định là 5.
        batch_size (int): Kích thước batch khi suy luận online. Mặc định là 200.
        shuffle (bool): Có xáo trộn dữ liệu hay không. Mặc định là False.

    Returns:
        DataLoader: DataLoader trả về từng batch (images, labels).
    """
    images, labels = load_corruption_arrays(data_root, corruption, severity)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def get_gradual_sequence(seed: int = 42) -> List[Tuple[str, int]]:
    """Tạo chuỗi (corruption, severity) tăng giảm dần cho thí nghiệm Gradual (Bảng 3).

    Mỗi corruption đi qua chuỗi độ khó 1 -> 2 -> 3 -> 4 -> 5 -> 4 -> 3 -> 2 -> 1
    theo thứ tự 15 loại corruption được xáo trộn ngẫu nhiên bởi seed.

    Parameters:
        seed (int): Seed cố định thứ tự xáo trộn các loại corruption.

    Returns:
        List[Tuple[str, int]]: Danh sách tuần tự các cặp (tên_corruption, severity).
    """
    rng = random.Random(seed)
    shuffled_corruptions = list(CORRUPTIONS_ORDER)
    rng.shuffle(shuffled_corruptions)

    severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    sequence: List[Tuple[str, int]] = []
    for corr in shuffled_corruptions:
        for sev in severities:
            sequence.append((corr, sev))
    return sequence