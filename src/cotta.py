"""Module cài đặt thuật toán Continual Test-Time Domain Adaptation (CoTTA).

File này chứa cài đặt từ đầu (reimplementation) của phương pháp CoTTA (CVPR 2022)
cùng với cơ chế cải tiến Gradient-Guided Restoration phục vụ phân tích phản biện.
"""

from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_model_for_cotta(model: nn.Module) -> nn.Module:
    """Cấu hình mô hình mạng nơ-ron để hoạt động ở chế độ Continual Test-Time Adaptation.

    Đặt mô hình vào chế độ train() và buộc các tầng BatchNorm tính toán thống kê
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
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


def copy_and_freeze_model(model: nn.Module) -> nn.Module:
    """Tạo một bản sao độc lập của mô hình và đóng băng toàn bộ gradient.

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
    """Lớp cài đặt thuật toán Continual Test-Time Domain Adaptation (CoTTA)."""

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
        anchor_model: Optional[nn.Module] = None,
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
            anchor_model (Optional[nn.Module]): Bản sao mô hình nguồn nguyên vẹn (eval mode).
        """
        super().__init__()
        if restoration_mode not in ("random", "gradient_guided"):
            raise ValueError(f"restoration_mode không hợp lệ: {restoration_mode}")
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

        # Teacher sử dụng cấu trúc giống Student (cập nhật EMA)
        self.teacher = copy_and_freeze_model(self.student)

        # Anchor model giữ nguyên trạng thái eval() và running_stats gốc từ nguồn
        if anchor_model is not None:
            self.anchor_model = anchor_model
        else:
            self.anchor_model = copy_and_freeze_model(self.student)

        self.source_Param_dict: Dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in self.student.named_parameters()
        }
        self.last_step_stats: Dict[str, float] = {}

    @staticmethod
    def compute_consistency_loss(
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        loss_type: str = "ce",
    ) -> torch.Tensor:
        """Tính hàm mất mát nhất quán giữa dự đoán của Student và nhãn giả của Teacher.

        Args:
            student_logits (torch.Tensor): Logits đầu ra của Student, kích thước (B, C).
            teacher_probs (torch.Tensor): Xác suất nhãn giả từ Teacher, kích thước (B, C).
            loss_type (str): "ce" hoặc "symmetric_ce".

        Returns:
            torch.Tensor: Giá trị hàm mất mát vô hướng trung bình trên batch.
        """
        student_log_probs = F.log_softmax(student_logits, dim=1)
        ce_teacher_to_student = -(teacher_probs * student_log_probs).sum(dim=1).mean()

        if loss_type == "ce":
            return ce_teacher_to_student

        student_probs = F.softmax(student_logits, dim=1)
        teacher_log_probs = torch.log(teacher_probs.clamp(min=1e-8))
        ce_student_to_teacher = -(student_probs * teacher_log_probs).sum(dim=1).mean()
        return 0.5 * (ce_teacher_to_student + ce_student_to_teacher)

    @torch.no_grad()
    def generate_pseudo_labels(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Sinh nhãn giả từ Teacher có kiểm tra ngưỡng tự tin trên mô hình nguồn f_{theta_0}.

        Args:
            x (torch.Tensor): Lô ảnh kiểm thử đầu vào tại bước t, kích thước (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, float, float]: (nhãn giả mềm, độ tự tin nguồn, cờ kích hoạt aug).
        """
        use_amp = x.is_cuda
        amp_ctx = (
            torch.amp.autocast("cuda", enabled=use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=use_amp)
        )

        with amp_ctx:
            anchor_logits = self.anchor_model(x).float()
            anchor_probs = F.softmax(anchor_logits, dim=1)
            max_confidences, _ = anchor_probs.max(dim=1)
            mean_source_confidence = float(max_confidences.mean().item())

            direct_teacher_probs = F.softmax(self.teacher(x).float(), dim=1)

            if self.augment_fn is None or self.num_augmentations <= 0:
                return direct_teacher_probs, mean_source_confidence, 0.0

            if mean_source_confidence >= self.confidence_threshold:
                return direct_teacher_probs, mean_source_confidence, 0.0

            aug_probs_sum = torch.zeros_like(direct_teacher_probs)
            for _ in range(self.num_augmentations):
                x_aug = self.augment_fn(x)
                aug_probs_sum += F.softmax(self.teacher(x_aug).float(), dim=1)

            aug_teacher_probs = aug_probs_sum / float(self.num_augmentations)
            return aug_teacher_probs, mean_source_confidence, 1.0

    @torch.no_grad()
    def update_teacher_ema(self) -> None:
        """Cập nhật trọng số Teacher bằng Exponential Moving Average (EMA)."""
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_param.data.mul_(self.ema_alpha).add_(
                student_param.data, alpha=1.0 - self.ema_alpha
            )

    @torch.no_grad()
    def apply_stochastic_restoration(self) -> None:
        """Khôi phục một tỷ lệ nhỏ (restoration_p) trọng số Student về trọng số nguồn."""
        if self.restoration_p <= 0.0:
            return

        for name, param in self.student.named_parameters():
            if not param.requires_grad or name not in self.source_Param_dict:
                continue

            source_weight = self.source_Param_dict[name]

            if self.restoration_mode == "random" or param.grad is None:
                mask = torch.empty_like(param.data).bernoulli_(self.restoration_p)
            else:
                grad_mag = param.grad.detach().abs().view(-1)
                num_elements = grad_mag.numel()
                k = max(1, int(round(self.restoration_p * num_elements)))

                if k >= num_elements:
                    mask = torch.ones_like(param.data)
                else:
                    _, smallest_indices = torch.topk(
                        grad_mag, k=k, largest=False, sorted=False
                    )
                    flat_mask = torch.zeros_like(grad_mag)
                    flat_mask.scatter_(0, smallest_indices, 1.0)
                    mask = flat_mask.view_as(param.data)

            param.data.copy_(mask * source_weight + (1.0 - mask) * param.data)

    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Thực hiện dự đoán trực tuyến và cập nhật thích nghi mô hình trên 1 test batch.

        Args:
            x (torch.Tensor): Lô ảnh kiểm thử không nhãn tại bước t, kích thước (B, C, H, W).

        Returns:
            torch.Tensor: Xác suất dự đoán đầu ra của Teacher cho lô ảnh hiện tại, (B, C).
        """
        self.optimizer.zero_grad()

        teacher_probs, mean_conf, aug_triggered = self.generate_pseudo_labels(x)
        student_logits = self.student(x)
        loss = self.compute_consistency_loss(
            student_logits=student_logits,
            teacher_probs=teacher_probs,
            loss_type=self.loss_type,
        )

        loss.backward()
        self.optimizer.step()
        self.update_teacher_ema()
        self.apply_stochastic_restoration()
        self.optimizer.zero_grad()

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


def build_cotta_from_config(
    model: nn.Module,
    cfg: Dict[str, object],
    augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_aug: bool = True,
    use_restore: bool = True,
    restoration_mode: str = "random",
    loss_type: str = "ce",
) -> CoTTA:
    """Khởi tạo đối tượng CoTTA từ cấu hình YAML, bảo toàn anchor_model nguồn ở eval().

    Args:
        model (nn.Module): Mô hình tiền huấn luyện trên miền nguồn.
        cfg (Dict[str, object]): Từ điển cấu hình đọc từ file YAML.
        augment_fn (Optional[Callable]): Hàm biến đổi ảnh từ src/augmentations.py.
        use_aug (bool): Bật/tắt Augmentation-Averaged Pseudo-Labels.
        use_restore (bool): Bật/tắt Stochastic Restoration.
        restoration_mode (str): "random" hoặc "gradient_guided".
        loss_type (str): "ce" hoặc "symmetric_ce".

    Returns:
        CoTTA: Đối tượng CoTTA sẵn sàng chạy online.
    """
    # QUAN TRỌNG: Sao chép và đóng băng anchor_model TRƯỚC khi configure_model_for_cotta()
    # xóa running_mean/running_var của Student!
    anchor_model = copy_and_freeze_model(model)
    anchor_model.eval()

    configured_model = configure_model_for_cotta(model)
    optimizer = torch.optim.Adam(
        configured_model.parameters(),
        lr=float(cfg.get("lr", 0.001)),
        betas=(0.9, 0.999),
    )

    effective_num_aug = int(cfg.get("num_aug", 32)) if use_aug else 0
    effective_restore_p = float(cfg.get("restore_p", 0.01)) if use_restore else 0.0

    return CoTTA(
        model=configured_model,
        optimizer=optimizer,
        augment_fn=augment_fn if use_aug else None,
        ema_alpha=float(cfg.get("alpha", 0.999)),
        restoration_p=effective_restore_p,
        confidence_threshold=float(cfg.get("p_th", 0.92)),
        num_augmentations=effective_num_aug,
        restoration_mode=restoration_mode,
        loss_type=loss_type,
        anchor_model=anchor_model,
    )