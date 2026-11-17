# Continual Test-Time Domain Adaptation (CoTTA) — Term Project

Kho mã nguồn thực nghiệm môn **Nhập môn Học máy (Introduction to Machine Learning)** của **Team 3 - Lớp 24KHMT1**.  
Thành viên nhóm 3:  
Nguyễn Minh Khôi - 24127066  
Võ Ngọc Thịnh - 24127246  
Lê Lâm Huy - 24127394

## 1. Thông tin Bài báo (Adopted Paper)
* **Tên bài báo:** Continual Test-Time Domain Adaptation
* **Nơi công bố:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022 (CORE A*)
* **Tác giả:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai
* **Official Link:** [CVPR 2022 Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
* **ArXiv Link:** [arXiv:2203.13591](https://arxiv.org/abs/2203.13591)

## 2. Lộ trình Thực hiện (Project Milestones)
- [x] **Milestone 1 — Selecting the Paper:** Đăng ký bài báo và khởi tạo kho mã nguồn công khai.
- [ ] **Milestone 2 & Lab 1 — Paper Reading & Experimental Reproduction:** Tự cài đặt lại thuật toán CoTTA (Weight-Averaged Teacher, Augmentation-Averaged Pseudo-labels, Stochastic Restoration) và tái lập thực nghiệm.
- [ ] **Milestone 3 & Lab 2 — Critical Analysis & Empirical Demonstration:** Phân tích điểm yếu lý thuyết của bài báo và thiết kế thực nghiệm chứng minh trên Poster.

## 3. Cấu trúc Kho mã nguồn
```text
├── README.md                        # Tài liệu tổng quan & Bảng ánh xạ W&B
├── requirements.txt                 # Danh sách thư viện phụ thuộc (Python >= 3.9)
├── configs/
│   ├── default.yaml                 # Cấu hình thực nghiệm CIFAR10-to-CIFAR10C
│   └── cifar100c.yaml               # Cấu hình thực nghiệm CIFAR100-to-CIFAR100C
├── src/
│   ├── __init__.py
│   ├── datasets.py                  # Tự động tải CIFAR10-C / CIFAR100-C từ Hugging Face
│   ├── utils.py                     # Cố định seed, tải source model, W&B helper
│   └── cotta.py                     # Tự cài đặt thuật toán CoTTA
├── experiments/                     # Kịch bản chạy tái lập (Lab 1) và phản biện (Lab 2)
└── notebooks/                       # Notebooks chạy Colab và trực quan hóa
```

## 4. Cài đặt Môi trường & Giải thích Thư viện
Yêu cầu **Python >= 3.9**.
```bash
git clone https://github.com/Thinh-MaNU/Machine-learning-project-.git
cd Machine-learning-project-
pip install -r requirements.txt
```

**Giải thích các thư viện ngoài danh sách chuẩn (`numpy`, `scipy`, `scikit-learn`, `torch`):**
- `torchvision`: Cung cấp các phép biến đổi ảnh tiêu chuẩn phục vụ Test-Time Augmentation cho CoTTA.
- `robustbench`: Dùng làm công cụ phụ trợ tải kiến trúc và trọng số source pre-trained chuẩn từ bài báo (WideResNet-28 cho CIFAR10, ResNeXt-29 cho CIFAR100).
- `huggingface_hub`: Tự động tải về và giải nén dữ liệu CIFAR10-C / CIFAR100-C trong lúc chạy (programmatic dataset loading).
- `wandb`: Lưu cấu hình, metrics và biểu đồ từng lượt chạy lên dự án công khai Weights & Biases (`huyle1156-hcmus/cotta-reproduction`).
- `pyyaml`, `tqdm`, `matplotlib`: Đọc file cấu hình `.yaml`, hiển thị thanh tiến trình và vẽ biểu đồ phân tích.

## 5. Hướng dẫn Chạy Thực nghiệm
*(Đang cập nhật theo tiến độ Lab 1)*

## 6. Bảng ánh xạ Thực nghiệm (W&B Mapping Table)
| Thực nghiệm | Script / Notebook | Link W&B Run | Tóm tắt kết quả |
| :--- | :--- | :--- | :--- |
| *(Đang cập nhật...)* | | | |                   # Notebooks chạy Colab và trực quan hóa