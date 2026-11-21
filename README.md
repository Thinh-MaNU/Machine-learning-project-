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
- [x] **Milestone 2 & Lab 1 — Paper Reading & Experimental Reproduction:** Tự cài đặt lại thuật toán CoTTA (Weight-Averaged Teacher, Augmentation-Averaged Pseudo-labels, Stochastic Restoration), tái lập thực nghiệm Core Scope, Sensitivity Sweep và đối chiếu mã nguồn gốc.
- [ ] **Milestone 3 & Lab 2 — Critical Analysis & Empirical Demonstration:** Phân tích điểm yếu lý thuyết của bài báo và thiết kế thực nghiệm chứng minh trên Poster.

## 3. Cấu trúc Kho mã nguồn
```text
├── README.md                        # Tài liệu tổng quan, hướng dẫn & Bảng ánh xạ W&B
├── plan.md                          # Kế hoạch phân công nhiệm vụ của nhóm
├── requirements.txt                 # Danh sách thư viện phụ thuộc (Python >= 3.9)
├── configs/
│   ├── default.yaml                 # Cấu hình thực nghiệm CIFAR10-to-CIFAR10C
│   └── cifar100c.yaml               # Cấu hình thực nghiệm CIFAR100-to-CIFAR100C
├── src/
│   ├── __init__.py
│   ├── datasets.py                  # Tự động tải CIFAR10-C / CIFAR100-C từ Hugging Face
│   ├── augmentations.py             # Tổ hợp 5 phép Test-Time Augmentation của CoTTA
│   ├── utils.py                     # Cố định seed, tải source model, W&B helper
│   └── cotta.py                     # Tự cài đặt thuật toán CoTTA (có cờ Ablation)
├── experiments/
│   ├── reproduce_cifar10c.py        # Tái lập Bảng 2 (CIFAR10-C & Ablation Study)
│   ├── reproduce_cifar100c.py       # Tái lập Bảng 4 (CIFAR100-C trên ResNeXt-29)
│   ├── sensitivity_sweep.py         # Khảo sát độ nhạy p_th, restore_p (Core Scope)
│   ├── compare_original.py          # Đối chiếu bản sạch vs mã nguồn gốc qinenergy/cotta
│   └── critique_ablation.py         # Kịch bản phản biện dành cho Mốc 3 & Lab 2
├── outputs/                         # Kết quả JSON và các biểu đồ PNG xuất cho Slide
└── notebooks/
    ├── colab_runner.ipynb           # Notebook chạy kiểm thử dữ liệu & Source check trên Colab
    └── plot_results.ipynb           # Notebook vẽ biểu đồ trực quan từ kết quả JSON
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
- `autoattack`: Gói phụ thuộc bắt buộc của `robustbench` khi nạp các mô hình chuẩn.

## 5. Hướng dẫn Chạy Thực nghiệm (Lab 1 — Core Scope)
Lưu ý: Bộ dữ liệu `CIFAR-10-C` (`YashP54/cifar10-c`) và `CIFAR-100-C` (`Huyle1156/cifar100-c`) sẽ tự động được tải về từ Hugging Face trong lúc chạy.

```bash
# 1. Tái lập Bảng 2: CIFAR10-to-CIFAR10C (Ablation: Weight-only, Weight+Aug, Full CoTTA)
python experiments/reproduce_cifar10c.py --config configs/default.yaml --seeds 42

# 2. Tái lập Bảng 4: CIFAR100-to-CIFAR100C (ResNeXt-29)
python experiments/reproduce_cifar100c.py --config configs/cifar100c.yaml --seed 42

# 3. Khảo sát độ nhạy siêu tham số (Sensitivity Sweep thay thế phần lược bỏ Core Scope)
python experiments/sensitivity_sweep.py --config configs/default.yaml --param p_th --values 0.75 0.98
python experiments/sensitivity_sweep.py --config configs/default.yaml --param restore_p --values 0.05 0.1

# 4. Đối chiếu bản sạch (CE theo Eq. 5) vs biến thể Symmetric CE (nhánh ImageNet của code gốc)
python experiments/compare_original.py --config configs/default.yaml --seed 42
```

## 6. Bảng ánh xạ Thực nghiệm (W&B Mapping Table)
Dự án W&B công khai: https://wandb.ai/huyle1156-hcmus/cotta-reproduction

| Thực nghiệm | Script / Notebook | Link W&B Run | Tóm tắt kết quả |
| :--- | :--- | :--- | :--- |
| CIFAR10-C Source Sanity Check | `notebooks/colab_runner.ipynb` | [cifar10-source-check](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/5m9wlthi) | Mean error 43.51% (khớp 43.5% Table 2) |
| CIFAR100-C Source Sanity Check | `notebooks/colab_runner.ipynb` | [cifar100-source-check](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/4qqig06d) | Mean error 46.45% (khớp 46.4% Table 4) |
| CIFAR10-C CoTTA (Weight-avg only) | `experiments/reproduce_cifar10c.py` | [weight_avg_only](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/91dbhath) | Mean error 18.34% (Paper: 18.30%, +0.20%) |
| CIFAR10-C CoTTA (Weight + Aug) | `experiments/reproduce_cifar10c.py` | [weight_and_aug](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/9m2w2ey8) | Mean error 19.33% (Paper: 17.40%, +11.08%) |
| CIFAR10-C CoTTA (Full, CE Eq. 5) | `experiments/reproduce_cifar10c.py` | [full_cotta_ce](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/8kit55s6) | Mean error 17.15% (Paper: 16.20%, +5.87%) |
| CIFAR100-C CoTTA (Full, ResNeXt-29) | `experiments/reproduce_cifar100c.py` | [cifar100c_full_cotta](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/5z82bzl5) | Mean error 31.37% (Paper: 32.50%, -3.49%) |
| Sensitivity Sweep (`p_th = 0.75`) | `experiments/sensitivity_sweep.py` | [sweep_p_th_0.75](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/9thhpvyb) | Mean error 17.49% (0.0% aug triggered) |
| Sensitivity Sweep (`p_th = 0.98`) | `experiments/sensitivity_sweep.py` | [sweep_p_th_0.98](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/zaoavopc) | Mean error 17.43% (99.07% aug triggered) |
| Sensitivity Sweep (`restore_p = 0.05`) | `experiments/sensitivity_sweep.py` | [sweep_restore_0.05](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/nlc5r5kr) | Mean error 17.42% (khôi phục hơi mạnh) |
| Sensitivity Sweep (`restore_p = 0.10`) | `experiments/sensitivity_sweep.py` | [sweep_restore_0.1](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/i73pmako) | Mean error 17.79% (cản trở thích nghi) |
| Đối chiếu hàm loss (Ours Symmetric CE) | `experiments/compare_original.py` | [ours_symmetric_ce](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/7go1891l) | Mean error 16.95% (+4.63% vs 16.20%) |
| Đối chiếu code gốc (`qinenergy/cotta`) | `Official Repo (Colab T4)` | [official_repo_cotta](https://wandb.ai/huyle1156-hcmus/cotta-reproduction/runs/9oygx2c5) | Mean error 16.42% (+1.36% vs 16.20%) |