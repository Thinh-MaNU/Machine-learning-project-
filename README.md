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

## 3. Cấu trúc Kho mã nguồn (Dự kiến)
```text
├── README.md                  # Tài liệu tổng quan dự án
├── requirements.txt           # Danh sách thư viện phụ thuộc
├── src/                       # Mã nguồn tự cài đặt lại thuật toán CoTTA
├── experiments/               # Kịch bản chạy tái lập (Reproduction) và phản biện (Critique)
└── notebooks/                 # Notebooks phân tích và trực quan hóa kết quả
