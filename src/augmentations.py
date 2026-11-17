"""Module cài đặt tổ hợp 5 phép biến đổi tăng cường dữ liệu (Test-Time Augmentation).

Cài đặt bám sát Mục 4.1 của bài báo CoTTA (CVPR 2022), bao gồm:
Color Jitter, Random Affine, Gaussian Blur, Random Horizontal Flip và Gaussian Noise.
"""

from typing import Callable

import torch
import torch.nn as nn
import torchvision.transforms as T


class AddGaussianNoise(nn.Module):
    """Cộng nhiễu Gaussian độc lập vào từng điểm ảnh của tensor trong miền [0, 1]."""

    def __init__(self, std: float = 0.005) -> None:
        """Khởi tạo lớp nhiễu Gaussian.

        Args:
            std (float): Độ lệch chuẩn của phân phối chuẩn sinh nhiễu (mặc định: 0.005).
        """
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Thực hiện cộng nhiễu Gaussian và giới hạn giá trị pixel về đoạn [0.0, 1.0].

        Args:
            x (torch.Tensor): Lô ảnh đầu vào kích thước (B, C, H, W).

        Returns:
            torch.Tensor: Lô ảnh sau khi cộng nhiễu Gaussian, kích thước (B, C, H, W).
        """
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


def build_cotta_augmenter(image_size: int = 32) -> Callable[[torch.Tensor], torch.Tensor]:
    """Khởi tạo chuỗi biến đổi ảnh ngẫu nhiên phục vụ Augmentation-Averaged Pseudo-Labels.

    Args:
        image_size (int): Kích thước cạnh ảnh vuông đầu vào (mặc định: 32 cho CIFAR-10/100).

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Hàm nhận batch tensor (B, C, H, W)
            và trả về batch ảnh đã qua biến đổi ngẫu nhiên.
    """
    pad_size = max(1, image_size // 4)
    return T.Compose([
        T.Pad(padding=pad_size, padding_mode="reflect"),
        T.RandomAffine(
            degrees=(-15, 15),
            translate=(0.0625, 0.0625),
            scale=(0.9, 1.1),
        ),
        T.RandomCrop(size=(image_size, image_size)),
        T.GaussianBlur(kernel_size=3, sigma=(0.001, 0.5)),
        T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        AddGaussianNoise(std=0.005),
    ])
