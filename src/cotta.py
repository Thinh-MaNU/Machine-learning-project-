"""Module cài đặt thuật toán Continual Test-Time Domain Adaptation (CoTTA).

File này chứa cài đặt từ đầu (reimplementation) của phương pháp CoTTA (CVPR 2022)
cùng với cơ chế cải tiến Gradient-Guided Restoration phục vụ phân tích phản biện.
"""

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_model_for_cotta(model: nn.Module) -> nn.Module:
    """Cấu hình mô hình mạng nơ-ron để hoạt động ở chế độ Continual Test-Time Adaptation.

    Đặt mô hình vào chế độ train() và buộc các tầng BatchNorm2d tính toán thống kê
    trung bình (mean) và phương sai (variance) trực tiếp trên từng test batch hiện tại
    thay vì sử dụng running stats của miền nguồn.

    Args:
        model (nn.Module): Mô hình tiền huấn luyện trên miền nguồn.

    Returns:
        nn.Module: Mô hình đã được cấu hình BatchNorm và bật gradient cho toàn bộ tham số.
    """
    model.train()
    model.requires_grad_(True)

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            # Buộc BatchNorm dùng thống kê (mu_B, sigma^2_B) của test batch hiện tại
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


def copy_and_freeze_model(model: nn.Module) -> nn.Module:
    """Tạo một bản sao độc lập của mô hình và đóng băng toàn bộ gradient.

    Dùng để khởi tạo mô hình nguồn cố định (Anchor model f_{theta_0}) và
    mô hình Teacher (f_{theta'_t}) không cập nhật qua lan truyền ngược.

    Args:
        model (nn.Module): Mô hình gốc cần sao chép.

    Returns:
        nn.Module: Bản sao của mô hình đã tắt requires_grad trên mọi tham số.
    """
    copied_model = deepcopy(model)
    for param in copied_model.parameters():
        param.detach_()
        param.requires_grad = False
    return copied_model


class CoTTA(nn.Module):
    """Lớp cài đặt thuật toán Continual Test-Time Domain Adaptation (CoTTA).

    Bao gồm 3 thành phần cốt lõi:
    1. Weight-Averaged Teacher với cập nhật Exponential Moving Average (EMA).
    2. Augmentation-Averaged Pseudo-Labels dựa trên ngưỡng tự tin của mô hình nguồn.
    3. Stochastic Restoration (hỗ trợ cả chế độ ngẫu nhiên Bernoulli gốc và
       chế độ Gradient-Guided phục vụ phản biện lý thuyết).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ema_alpha: float = 0.999,
        restoration_p: float = 0.01,
        confidence_threshold: float = 0.92,
        num_augmentations: int = 32,
        restoration_mode: str = "random",
        loss_type: str = "ce",
    ) -> None:
        """Khởi tạo bộ thích nghi trực tuyến CoTTA.

        Args:
            model (nn.Module): Mô hình Student đã cấu hình qua configure_model_for_cotta().
            optimizer (torch.optim.Optimizer): Bộ tối ưu hóa cập nhật trọng số Student.
            augment_fn (Optional[Callable]): Hàm biến đổi tăng cường dữ liệu ảnh ngẫu nhiên.
            ema_alpha (float): Hệ số làm mượt EMA cho Teacher (mặc định: 0.999).
            restoration_p (float): Tỷ lệ khôi phục trọng số về mô hình nguồn (mặc định: 0.01).
            confidence_threshold (float): Ngưỡng tự tin p_th để kích hoạt N phép tăng cường.
            num_augmentations (int): Số lượng phép tăng cường N khi độ tự tin thấp (mặc định: 32).
            restoration_mode (str): Chế độ khôi phục trọng số ("random" hoặc "gradient_guided").
            loss_type (str): Hàm mất mát nhất quán ("ce" hoặc "symmetric_ce").
        """
        super().__init__()
        if restoration_mode not in ("random", "gradient_guided"):
            raise ValueError(
                f"restoration_mode không hợp lệ: {restoration_mode}")
        if loss_type not in ("ce", "symmetric_ce"):
            raise ValueError(f"loss_type không hợp lệ: {loss_type}")

        self.student = model
        self.optimizer = optimizer
        self.augment_fn = augment_fn
        self.ema_alpha = ema_alpha
        self.restoration_p = restoration_p
        self.confidence_threshold = confidence_threshold
        self.num_augmentations = num_augmentations
        self.restoration_mode = restoration_mode
        self.loss_type = loss_type

        # Khởi tạo Teacher (theta'_0 = theta_0) và Mô hình nguồn cố định (theta_0)
        self.teacher = copy_and_freeze_model(self.student)
        self.anchor_model = copy_and_freeze_model(self.student)

        # Lưu trữ bản sao state_dict của trọng số nguồn theta_0 để khôi phục nhanh
        self.source_Param_dict: Dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in self.student.named_parameters()
        }

        # Thống kê theo dõi phục vụ phân tích thực nghiệm
        self.last_step_stats: Dict[str, float] = {}

    @staticmethod
    def compute_consistency_loss(
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        loss_type: str = "ce",
    ) -> torch.Tensor:
        """Tính hàm mất mát nhất quán giữa dự đoán của Student và nhãn giả của Teacher.

        Args:
            student_logits (torch.Tensor): Logits đầu ra của mạng Student, kích thước (B, C).
            teacher_probs (torch.Tensor): Phân phối xác suất nhãn giả từ Teacher, kích thước (B, C).
            loss_type (str): "ce" (Cross-Entropy mềm chuẩn theo Eq. 2 trong bài báo) hoặc
                "symmetric_ce" (Cross-Entropy đối xứng dùng trong một số nhánh code gốc).

        Returns:
            torch.Tensor: Giá trị hàm mất mát vô hướng (scalar tensor) trung bình trên batch.
        """
        student_log_probs = F.log_softmax(student_logits, dim=1)
        ce_teacher_to_student = - \
            (teacher_probs * student_log_probs).sum(dim=1).mean()

        if loss_type == "ce":
            return ce_teacher_to_student

        student_probs = F.softmax(student_logits, dim=1)
        teacher_log_probs = torch.log(teacher_probs.clamp(min=1e-8))
        ce_student_to_teacher = - \
            (student_probs * teacher_log_probs).sum(dim=1).mean()
        return 0.5 * (ce_teacher_to_student + ce_student_to_teacher)

    @torch.no_grad()
    def generate_pseudo_labels(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Sinh nhãn giả (pseudo-labels) từ Teacher có kết hợp kiểm tra ngưỡng tự tin nguồn.

        Nếu độ tự tin trung bình của mô hình nguồn f_{theta_0}(x) < p_th, hệ thống
        lấy trung bình dự đoán của Teacher qua N phép biến đổi tăng cường ảnh.

        Args:
            x (torch.Tensor): Lô ảnh kiểm thử đầu vào tại bước t, kích thước (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, float, float]:
                - Nhãn giả xác suất mềm từ Teacher, kích thước (B, num_classes).
                - Độ tự tin trung bình của mô hình nguồn trên batch hiện tại.
                - Tỷ lệ mẫu (0.0 hoặc 1.0) đã kích hoạt N phép tăng cường dữ liệu.
        """
        # 1. Tính độ tự tin trên mô hình nguồn đóng băng f_{theta_0}
        anchor_logits = self.anchor_model(x)
        anchor_probs = F.softmax(anchor_logits, dim=1)
        max_confidences, _ = anchor_probs.max(dim=1)
        mean_source_confidence = float(max_confidences.mean().item())

        # 2. Dự đoán trực tiếp của Teacher trên ảnh gốc
        direct_teacher_probs = F.softmax(self.teacher(x), dim=1)

        # Nếu không có hàm augmentation hoặc độ tự tin trung bình đạt ngưỡng p_th
        if self.augment_fn is None or self.num_augmentations <= 0:
            return direct_teacher_probs, mean_source_confidence, 0.0

        if mean_source_confidence >= self.confidence_threshold:
            return direct_teacher_probs, mean_source_confidence, 0.0

        # 3. Khi độ tự tin < p_th: Lấy trung bình dự đoán qua N phép tăng cường
        aug_probs_sum = torch.zeros_like(direct_teacher_probs)
        for _ in range(self.num_augmentations):
            x_aug = self.augment_fn(x)
            aug_probs_sum += F.softmax(self.teacher(x_aug), dim=1)

        aug_teacher_probs = aug_probs_sum / float(self.num_augmentations)
        return aug_teacher_probs, mean_source_confidence, 1.0

    @torch.no_grad()
    def update_teacher_ema(self) -> None:
        """Cập nhật trọng số Teacher bằng trung bình trượt hàm mũ (EMA).

        Công thức: theta'_{t+1} = alpha * theta'_t + (1 - alpha) * theta_{t+1}.
        """
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_param.data.mul_(self.ema_alpha).add_(
                student_param.data, alpha=1.0 - self.ema_alpha
            )

    @torch.no_grad()
    def apply_stochastic_restoration(self) -> None:
        """Khôi phục một tỷ lệ nhỏ (restoration_p) trọng số của Student về trọng số nguồn.

        Hỗ trợ 2 chế độ:
        - "random": Lấy mẫu mặt nạ M ~ Bernoulli(p) đồng nhất (CoTTA chuẩn).
        - "gradient_guided": Khôi phục đúng tỷ lệ p các trọng số có độ lớn gradient
          |grad| nhỏ nhất ở bước hiện tại (bảo toàn các trọng số đang thích nghi mạnh).
        """
        if self.restoration_p <= 0.0:
            return

        for name, param in self.student.named_parameters():
            if not param.requires_grad or name not in self.source_Param_dict:
                continue

            source_weight = self.source_Param_dict[name]

            if self.restoration_mode == "random" or param.grad is None:
                # Chế độ gốc: Mặt nạ nhị phân M ~ Bernoulli(p)
                mask = torch.empty_like(
                    param.data).bernoulli_(self.restoration_p)
            else:
                # Chế độ cải tiến (Critique 1): Khôi phục p% trọng số có |grad| nhỏ nhất
                grad_mag = param.grad.detach().abs().view(-1)
                num_elements = grad_mag.numel()
                k = max(1, int(round(self.restoration_p * num_elements)))

                if k >= num_elements:
                    mask = torch.ones_like(param.data)
                else:
                    # Tìm k vị trí có độ lớn đạo hàm nhỏ nhất
                    _, smallest_indices = torch.topk(
                        grad_mag, k=k, largest=False, sorted=False
                    )
                    flat_mask = torch.zeros_like(grad_mag)
                    flat_mask.scatter_(0, smallest_indices, 1.0)
                    mask = flat_mask.view_as(param.data)

            # Cập nhật: theta_{t+1} <- M * theta_0 + (1 - M) * theta_{t+1}
            param.data.copy_(mask * source_weight + (1.0 - mask) * param.data)

    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Thực hiện dự đoán trực tuyến và cập nhật thích nghi mô hình trên 1 test batch.

        Args:
            x (torch.Tensor): Lô ảnh kiểm thử không nhãn tại bước t, kích thước (B, C, H, W).

        Returns:
            torch.Tensor: Xác suất dự đoán đầu ra của Teacher cho lô ảnh hiện tại, (B, num_classes).
        """
        self.optimizer.zero_grad()

        # Bước 1: Sinh nhãn giả từ Teacher (kèm kiểm tra ngưỡng p_th)
        teacher_probs, mean_conf, aug_triggered = self.generate_pseudo_labels(
            x)

        # Bước 2: Lan truyền tiến qua Student và tính Consistency Loss
        student_logits = self.student(x)
        loss = self.compute_consistency_loss(
            student_logits=student_logits,
            teacher_probs=teacher_probs,
            loss_type=self.loss_type,
        )

        # Bước 3: Lan truyền ngược và cập nhật trọng số Student
        loss.backward()
        self.optimizer.step()

        # Bước 4: Cập nhật EMA cho Teacher
        self.update_teacher_ema()

        # Bước 5: Khôi phục trọng số (Stochastic / Gradient-Guided Restoration)
        # Lưu ý: Gọi trước khi xóa grad để chế độ gradient_guided sử dụng được param.grad
        self.apply_stochastic_restoration()
        self.optimizer.zero_grad()

        # Lưu thống kê bước hiện tại để log lên W&B
        self.last_step_stats = {
            "loss": float(loss.item()),
            "source_confidence": mean_conf,
            "aug_triggered": aug_triggered,
        }

        return teacher_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hàm giao diện chuẩn của nn.Module gọi trực tiếp forward_and_adapt.

        Args:
            x (torch.Tensor): Lô ảnh kiểm thử đầu vào, kích thước (B, C, H, W).

        Returns:
            torch.Tensor: Dự đoán xác suất từ Teacher, kích thước (B, num_classes).
        """
        return self.forward_and_adapt(x)
