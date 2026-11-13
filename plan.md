# TÀI LIỆU HƯỚNG DẪN & KẾ HOẠCH THỰC HIỆN ĐỒ ÁN NHẬP MÔN HỌC MÁY
**Đề tài:** Continual Test-Time Domain Adaptation (CoTTA — CVPR 2022)  
**Quy mô nhóm:** 3 Thành viên  

---

## 1. THÔNG TIN BÀI BÁO & ĐĂNG KÝ (SELECTION FORM)

Dưới đây là thông tin chuẩn hóa của bài báo đã được kiểm duyệt theo quy chuẩn **CORE $A^*$** của môn học:

* **Publication title:** Continual Test-Time Domain Adaptation
* **Publication year:** 2022
* **Publication month:** 6
* **Publication venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
* **Official link:** https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html
* **Open-access link:** https://arxiv.org/abs/2203.13591
* **Quality of the publication:** $A^*$
* **Proof of quality:** https://portal.core.edu.au/conf-ranks/604/

---

## 2. QUY CHẾ CỐT LÕI & CƠ CẤU ĐIỂM SỐ CỦA MÔN HỌC

Cả nhóm cần nắm vững cơ cấu điểm và các quy tắc liêm chính học thuật để tránh mất điểm đáng tiếc:

### 2.1. Bảng trọng số điểm
| Hạng mục đánh giá | Trọng số | Yêu cầu cốt lõi |
| :--- | :--- | :--- |
| **1. Selection Form** | Điều kiện qua môn (Pass gate) | Đăng ký bài báo và nộp link GitHub Repo công khai. Không đăng ký sẽ nhận $0$ điểm toàn đồ án. |
| **2. Slide Deck** (Paper Reading + Experimental Reproduction) | **$45\%$** | Trình bày lại bài báo bằng ngôn từ và hình vẽ tự thiết kế của nhóm; tự code lại thuật toán từ đầu; đối chiếu kết quả chạy lại với số liệu gốc và giải thích mọi độ lệch. |
| **3. Poster** (Critical Analysis + Empirical Demonstration) | **$50\%$** | Chỉ ra điểm yếu thực sự trong lý thuyết (giả định, độ phức tạp) của bài báo và tự thiết kế thí nghiệm để chứng minh các điểm yếu đó bằng số liệu thực tế. |
| **4. Kỷ luật Quản lý Repo, W&B & Định dạng** | **$5\%$** | Lịch sử commit đều đặn từ tuần đầu tiên trên GitHub; mọi con số báo cáo đều gắn link tới lượt chạy trên **Weights & Biases (W&B)**; định dạng Slide/Poster chỉn chu. |
| **Tổng cộng** | **$100\%$** | Thi vấn đáp cuối kỳ dưới hình thức **Poster Workshop**. |

### 2.2. Nguyên tắc bắt buộc (Quan trọng)
1. **Tự cài đặt lại (Reimplement from scratch):** Nhóm bắt buộc phải tự viết code thuật toán dựa trên mô tả lý thuyết của bài báo. Code công khai của tác giả chỉ được phép dùng để tham khảo kiểm chứng và so sánh kết quả.
2. **Nghiêm cấm sao chép:** Không copy văn bản, không chụp màn hình hình vẽ/sơ đồ của bài báo gốc đưa vào Slide/Poster, không copy code gốc hoặc code của nhóm khác (vi phạm sẽ bị $0$ điểm đồ án).
3. **Truy vết thực nghiệm (GitHub & W&B):** Giảng viên đánh giá tiến độ qua lịch sử commit hàng tuần trên GitHub. Mọi bảng biểu, con số xuất hiện trên Slide Deck và Poster đều phải liên kết trực tiếp tới một lượt chạy (run) trên dự án W&B công khai của nhóm.
4. **Trách nhiệm tại buổi bảo vệ Poster Workshop:** Cả 3 thành viên cùng đứng cạnh Poster để trình bày và trả lời câu hỏi. Giảng viên có thể hỏi bất kỳ thành viên nào về bất kỳ phần nào của đồ án (lý thuyết, code, thực nghiệm).

---

## 3. TÓM TẮT LÝ THUYẾT & TOÁN HỌC CỐT LÕI CỦA CoTTA

### 3.1. Phát biểu bài toán (Continual Test-Time Adaptation)
Cho một mô hình phân lớp ảnh đã được tiền huấn luyện trên miền nguồn (Source domain) với bộ tham số $\theta_0$, ký hiệu là $f_{\theta_0}$.  
Trong giai đoạn kiểm thử (Test time), mô hình tiếp nhận một luồng dữ liệu không nhãn $x_t$ đến liên tục theo thời gian $t$ từ một chuỗi các miền đích biến đổi không ngừng:
$$\mathcal{D}_1 \to \mathcal{D}_2 \to \dots \to \mathcal{D}_K$$
Tại mỗi bước $t$, mô hình vừa phải đưa ra dự đoán $\hat{y}_t$ cho batch hiện tại $x_t$, vừa phải tự cập nhật tham số $\theta_t \to \theta_{t+1}$ một cách trực tuyến (online, không nhãn) trước khi batch $x_{t+1}$ tiếp theo xuất hiện.

Hai thách thức lớn nhất của bài toán này là:
1. **Tích lũy sai số (Error Accumulation):** Nhãn giả (pseudo-labels) tự sinh ra bị nhiễu khi miền dữ liệu thay đổi, nếu dùng để tự huấn luyện lâu dài sẽ làm mô hình lệch dần.
2. **Quên thảm khốc (Catastrophic Forgetting):** Sau khi thích nghi qua nhiều miền nhiễu liên tiếp, mô hình đánh mất tri thức gốc $\theta_0$ và suy giảm khả năng tổng quát hóa trên các miền mới.

### 3.2. Ba thành phần thuật toán của CoTTA

#### Thành phần 1: Weight-Averaged Teacher và Consistency Loss
* Khởi tạo cả mô hình **Student** ($\theta_0$) và mô hình **Teacher** ($\theta'_0 = \theta_0$) bằng trọng số nguồn ban đầu.
* Tại bước $t$, Teacher tạo ra dự đoán nhãn giả $\tilde{y}'_t$. Student cập nhật tham số $\theta_t$ bằng Gradient Descent để tối thiểu hóa hàm mất mát Cross-Entropy đối xứng giữa dự đoán của Student $\hat{y}_t = f_{\theta_t}(x_t)$ và nhãn giả của Teacher:
  $$\mathcal{L}_{\text{CoTTA}}(\theta_t; x_t) = -\frac{1}{C} \sum_{c=1}^{C} \tilde{y}'_{t,c} \log \hat{y}_{t,c}$$
  trong đó $C$ là tổng số lớp phân loại.
* Ngay sau khi Student cập nhật xong $\theta_{t+1}$, trọng số của Teacher được cập nhật mượt bằng trung bình trượt hàm mũ (**Exponential Moving Average — EMA**) với hệ số $\alpha = 0.999$:
  $$\theta'_{t+1} = \alpha \theta'_t + (1 - \alpha) \theta_{t+1}$$

#### Thành phần 2: Augmentation-Averaged Pseudo-Labels (Lọc nhiễu nhãn giả)
* Khi miền dữ liệu dịch chuyển đột ngột, dự đoán đơn lẻ của Teacher cũng có thể bị nhiễu. CoTTA dùng độ tự tin của mô hình nguồn cố định $f_{\theta_0}(x_t)$ để đo khoảng cách dịch chuyển miền (domain gap).
* Nếu độ tự tin $\max_c f_{\theta_0}(x_t)_c$ lớn hơn hoặc bằng ngưỡng $p_{\text{th}}$ (mặc định $p_{\text{th}} = 0.92$ trên CIFAR-10-C), tức là miền dịch chuyển nhẹ, ta dùng trực tiếp dự đoán của Teacher.
* Ngược lại, nếu $\max_c f_{\theta_0}(x_t)_c < p_{\text{th}}$ (miền dịch chuyển nặng), ta lấy trung bình dự đoán của Teacher qua $N = 32$ phép biến đổi tăng cường dữ liệu ngẫu nhiên $\text{aug}_i(x_t)$ (gồm Random Crop, Horizontal Flip, ColorJitter, Gaussian Noise, Gaussian Blur, Rotation...):
  $$\tilde{y}'_t = \begin{cases} f_{\theta'_t}(x_t) & \text{nếu } \max_c f_{\theta_0}(x_t)_c \ge p_{\text{th}} \\ \frac{1}{N} \sum_{i=1}^{N} f_{\theta'_t}(\text{aug}_i(x_t)) & \text{ngược lại} \end{cases}$$
* Dự đoán cuối cùng xuất ra tại thời điểm kiểm thử cũng chính là $\tilde{y}'_t$ của Teacher.

#### Thành phần 3: Stochastic Restoration (Khôi phục ngẫu nhiên trọng số để chống quên)
* Sau bước cập nhật gradient của Student ($\theta_{t+1}$), để ngăn mạng trôi quá xa khỏi tri thức tiền huấn luyện ban đầu, CoTTA lấy mẫu một mặt nạ nhị phân $M$ có cùng kích thước với tensor trọng số theo phân phối Bernoulli với xác suất rất nhỏ $p = 0.01$:
  $$M \sim \text{Bernoulli}(p)$$
  $$\theta_{t+1} \leftarrow M \odot \theta_0 + (1 - M) \odot \theta_{t+1}$$
  trong đó $\odot$ là phép nhân từng phần tử (element-wise multiplication).
* Như vậy, tại mỗi bước thích nghi, $1\%$ trọng số được chọn ngẫu nhiên của mạng Student sẽ được khôi phục trực tiếp về giá trị gốc $\theta_0$.

---

## 4. HƯỚNG DẪN GIAI ĐOẠN 1: TÁI LẬP THỰC NGHIỆM (SLIDE DECK — $45\%$)

### 4.1. Tập dữ liệu và Mô hình chuẩn bị
* **Tập dữ liệu ưu tiên số 1:** **CIFAR-10-C** và **CIFAR-100-C**. Mỗi tập gồm $15$ loại nhiễu ảnh khác nhau (Gaussian Noise, Shot Noise, Impulse Noise, Defocus Blur, Glass Blur, Motion Blur, Zoom Blur, Snow, Frost, Fog, Brightness, Contrast, Elastic Transform, Pixelate, JPEG Compression).
* **Thiết lập Continual Benchmark:** Ghép nối tiếp $15$ loại nhiễu ở mức độ biến dạng nặng nhất (**Severity Level 5**). Mỗi loại nhiễu có $10.000$ ảnh test $\to$ Tổng cộng luồng dữ liệu gồm $150.000$ ảnh đi qua mô hình liên tiếp không được reset trọng số.
* **Trọng số tiền huấn luyện (Pre-trained weights):** Tải trực tiếp từ thư viện `RobustBench` (không cần tốn thời gian huấn luyện từ đầu):
  * CIFAR-10-C: Kiến trúc `WideResNet-28-10` (Standard).
  * CIFAR-100-C: Kiến trúc `ResNeXt-29_32x4d`.
* **Thời gian chạy dự kiến:** Khoảng $10 - 15$ phút cho 1 lượt chạy qua toàn bộ $15$ miền trên 1 GPU (Google Colab T4 hoặc Kaggle P100).

### 4.2. Các phương pháp đối chứng (Baselines) cần chạy để so sánh
1. **Source-only:** Dùng thẳng mô hình gốc $f_{\theta_0}$ dự đoán mà không cập nhật gì (Error Rate trên CIFAR-10-C khoảng $43.5\%$).
2. **BN-Adapt (Test-time BatchNorm):** Chỉ cập nhật kỳ vọng $\mu_B$ và phương sai $\sigma^2_B$ của các tầng BatchNorm trên từng batch test mà không cập nhật trọng số (Error Rate khoảng $20.4\%$).
3. **TENT-continual:** Cập nhật tham số affine $(\gamma, \beta)$ của tầng BatchNorm bằng cực tiểu hóa Entropy liên tục (sẽ bị tích lũy sai số dần qua các miền sau).
4. **CoTTA (Bản nhóm tự cài đặt):** Mục tiêu đạt trung bình Error Rate tiệm cận con số bài báo công bố ($\approx 16.2\%$ trên CIFAR-10-C và $\approx 32.5\%$ trên CIFAR-100-C).

### 4.3. Phân tích độ lệch (Deviation Analysis) cho Slide Deck
Khi số liệu nhóm chạy ra chênh lệch nhỏ (ví dụ $\pm 0.2\% \to 0.5\%$) so với bảng trong bài báo gốc, nhóm cần ghi nhận và giải thích rõ trong Slide dựa trên 3 yếu tố:
* Sự ngẫu nhiên của hạt giống (`random seed`) trong hai quá trình: lấy mẫu mặt nạ $M \sim \text{Bernoulli}(0.01)$ và lấy mẫu $N=32$ phép tăng cường ảnh ngẫu nhiên ở mỗi batch.
* Khác biệt nhỏ trong cài đặt hàm biến đổi ảnh (như `GaussianBlur`, `ColorJitter`, chế độ nội suy `interpolation` giữa các phiên bản `torchvision`).
* Sai số dấu chấm động trên các kiến trúc phần cứng GPU/CUDA khác nhau.

---

## 5. HƯỚNG DẪN GIAI ĐOẠN 2: PHẢN BIỆN LÝ THUYẾT & THỰC NGHIỆM (POSTER — $50\%$)

Để đạt điểm tối đa ở Poster, nhóm cần trình bày **3 điểm yếu lý thuyết** kèm **3 bộ thí nghiệm tự thiết kế** dưới đây:

### Điểm yếu 1: Sự "mù quáng" của giả định phân phối đồng nhất trong Stochastic Restoration
* **Phản biện lý thuyết (Theoretical Critique):**  
  CoTTA dùng mặt nạ $M \sim \text{Bernoulli}(p)$ với xác suất đồng nhất $p = 0.01$ cho mọi tham số trong mạng. Xét theo lý thuyết tối ưu hóa và hình học thông tin (Information Geometry), độ nhạy của hàm mục tiêu đối với từng trọng số là hoàn toàn khác nhau: các trọng số có đạo hàm $|\nabla_\theta \mathcal{L}|$ lớn đang đóng vai trò quyết định giúp mô hình thích nghi với đặc trưng của miền đích hiện tại, trong khi các trọng số có $|\nabla_\theta \mathcal{L}| \approx 0$ ít đóng góp nhưng dễ bị trôi nhiễu. Việc khôi phục ngẫu nhiên đồng nhất (Uniform Restoration) sẽ vô tình xóa đi các trọng số quan trọng vừa học được và gây phương sai lớn nếu $p$ thay đổi.
* **Thí nghiệm chứng minh (Empirical Demonstration):**
  1. **Khảo sát độ nhạy tham số $p$:** Chạy CoTTA với $p \in \{0.0, 0.001, 0.005, 0.01, 0.05, 0.1\}$ qua nhiều `seed` để chứng minh mô hình sụp đổ nhanh khi $p \ge 0.05$.
  2. **Đề xuất cải tiến *Gradient-Guided Restoration*:** Thay vì chọn ngẫu nhiên $1\%$ trọng số, ta ưu tiên khôi phục $1\%$ các trọng số có độ lớn gradient $|\nabla_\theta \mathcal{L}|$ nhỏ nhất (hoặc kết hợp xác suất tỷ lệ nghịch với độ lớn gradient). Vẽ biểu đồ so sánh Error Rate giữa *Random Restoration* và *Guided Restoration* trên Poster.

### Điểm yếu 2: Giả định Batch dữ liệu lớn & độc lập đồng phân phối (i.i.d. Assumption)
* **Phản biện lý thuyết (Theoretical Critique):**  
  Trong CoTTA, toàn bộ các tầng Batch Normalization (BN) không dùng thống kê $(\mu_0, \sigma^2_0)$ của miền nguồn mà tính trực tiếp trung bình $\mu_B$ và phương sai $\sigma^2_B$ trên từng batch kiểm thử $B$ hiện tại (bài báo cố định kích thước batch rất lớn $B = 200$ và trộn đều nhãn lớp). Giả định ngầm định này vi phạm thực tế triển khai trên hệ thống thị giác động (như robot hoặc xe tự hành):
  1. Dữ liệu đến từng khung hình hoặc lô rất nhỏ ($B = 1, 2, 4$), khiến ước lượng phương sai mẫu $\sigma^2_B$ có phương sai cực lớn và làm lệch chuẩn toàn bộ biểu diễn qua các tầng sâu.
  2. Dữ liệu thực tế có tính tương quan nhãn theo thời gian (**Non-i.i.d. / Temporal Label Correlation** — ví dụ trong một khoảng thời gian chỉ xuất hiện 1–2 lớp đối tượng). Khi một batch chỉ chứa một vài lớp, hàm mất mát Cross-Entropy đẩy Teacher dự đoán lệch hẳn về lớp chiếm đa số, gây sụp đổ mô hình (**Model Collapse**).
* **Thí nghiệm chứng minh (Empirical Demonstration):**
  1. **Thí nghiệm thu nhỏ Batch Size:** Giữ nguyên thuật toán CoTTA và giảm dần kích thước batch kiểm thử $B \in \{200, 64, 16, 4, 2, 1\}$ trên CIFAR-10-C. Vẽ đồ thị Error Rate tăng vọt khi $B \le 16$.
  2. **Thí nghiệm luồng dữ liệu Non-i.i.d. (Dirichlet Stream):** Sắp xếp lại các mẫu trong từng miền nhiễu của CIFAR-10-C theo phân phối Dirichlet $\text{Dir}(\delta)$ với hệ số tập trung $\delta \in \{10.0, 1.0, 0.1, 0.01\}$ (khi $\delta$ càng nhỏ, các ảnh cùng lớp càng tụ lại thành từng cụm thời gian). Chứng minh bằng biểu đồ rằng sai số của CoTTA tăng vọt từ $\sim 16.2\%$ lên trên $60\%$ khi $\delta = 0.01$.

### Điểm yếu 3: Nút cổ chai độ phức tạp tính toán $\mathcal{O}(N)$ và Ngưỡng tự tin $p_{\text{th}}$ thiếu hiệu chỉnh
* **Phản biện lý thuyết (Theoretical Critique):**  
  * *Về độ phức tạp (Complexity Story):* Mỗi khi một batch có độ tự tin trung bình dưới ngưỡng $p_{\text{th}}$, CoTTA phải thực hiện $N = 32$ lượt lan truyền tiến (forward passes) qua mô hình Teacher, cộng thêm $1$ lượt forward qua mô hình nguồn $f_{\theta_0}$, $1$ lượt forward và $1$ lượt lan truyền ngược (backward pass) qua Student $f_{\theta_t}$. Tổng chi phí tính toán lúc suy luận tăng gấp $\sim 35$ lần so với suy luận thông thường, khiến thuật toán không thể đáp ứng yêu cầu thời gian thực (real-time FPS).
  * *Về hiệu chỉnh xác suất (Calibration):* Việc dùng xác suất Softmax $\max_c f_{\theta_0}(x_t)_c$ để so sánh với ngưỡng cố định $p_{\text{th}}$ bỏ qua hiện tượng **Overconfidence** của mạng nơ-ron sâu trên dữ liệu ngoài phân phối (Out-of-Distribution), dẫn đến việc kích hoạt hoặc bỏ qua bước tăng cường dữ liệu không chính xác ở một số loại nhiễu.
* **Thí nghiệm chứng minh (Empirical Demonstration):**
  1. **Đường cong đánh đổi Hiệu năng – Độ trễ (Accuracy vs. Latency Trade-off):** Chạy CoTTA với số lượng phép tăng cường $N \in \{1, 2, 4, 8, 16, 32\}$, đo thời gian xử lý trung bình trên mỗi batch ($\text{ms/batch}$) và tổng thời gian chạy. Chứng minh rằng từ $N = 8$ lên $N = 32$, độ chính xác chỉ tăng dưới $0.4\%$ nhưng thời gian suy luận tăng gấp $4$ lần.
  2. **Phân tích tần suất kích hoạt $p_{\text{th}}$:** Thống kê tỷ lệ $\%$ số batch kích hoạt $32$ phép tăng cường trên từng loại nhiễu trong $15$ nhiễu và đối chiếu với sai số thực tế của mô hình nguồn $f_{\theta_0}$ để chỉ ra sự thiếu hiệu chỉnh (miscalibration).

---

## 6. PHÂN CÔNG NHIỆM VỤ & LỘ TRÌNH 12 TUẦN CHO NHÓM 3 NGƯỜI

### 6.1. Phân vai chuyên trách (Đảm bảo ai cũng có code và phần trình bày riêng)
* **Thành viên 1 — Core Algorithm & Optimization Lead:**
  * *Code tái lập:* Tự cài đặt kiến trúc *Weight-Averaged Teacher (EMA)*, hàm mất mát $\mathcal{L}_{\text{CoTTA}}$ và cơ chế *Stochastic Restoration* trong `src/cotta.py`.
  * *Slide & Poster:* Phụ trách vẽ sơ đồ kiến trúc tổng quan cho Slide và thực hiện trọn vẹn **Điểm yếu 1** (Thí nghiệm độ nhạy $p$ & *Gradient-Guided Restoration*).
* **Thành viên 2 — Data Stream & W&B Infrastructure Lead:**
  * *Code tái lập:* Xây dựng pipeline tải dữ liệu liên tục CIFAR-10-C / CIFAR-100-C (`src/datasets.py`), tổ hợp $32$ phép biến đổi ảnh (`src/augmentations.py`), và quản lý dự án **Weights & Biases (W&B)**.
  * *Slide & Poster:* Phụ trách trực quan hóa luồng dữ liệu, xuất biểu đồ tái lập từ W&B cho Slide và thực hiện trọn vẹn **Điểm yếu 2** (Thí nghiệm Batch Size nhỏ & Luồng dữ liệu Non-i.i.d. Dirichlet).
* **Thành viên 3 — Baselines, Deviation & Complexity Lead:**
  * *Code tái lập:* Cài đặt các phương pháp đối chứng (*Source-only*, *BN-Adapt*, *TENT*), viết script đánh giá tổng hợp và phân tích nguyên nhân sai lệch số liệu (Deviation Analysis).
  * *Slide & Poster:* Phụ trách bảng so sánh kết quả tái lập trong Slide, thiết kế bố cục đồ họa cho Poster và thực hiện trọn vẹn **Điểm yếu 3** (Thí nghiệm độ phức tạp $\mathcal{O}(N)$, thời gian suy luận & ngưỡng $p_{\text{th}}$).

### 6.2. Tiến độ chi tiết từng tuần
| Tuần | Mục tiêu trọng tâm | Việc cần hoàn thành & Commit lên GitHub |
| :--- | :--- | :--- |
| **Tuần 1** | **Khởi động & Đăng ký (Milestone 1)** | Nộp Selection Form trên Moodle; khởi tạo Public GitHub Repo; tạo tài khoản & Public Project trên `wandb.ai`; cả 3 thành viên accept lời mời vào GitHub Repo. |
| **Tuần 2** | **Đọc hiểu sâu & Thiết lập Dữ liệu** | Cả nhóm đọc kỹ bài báo CoTTA; Thành viên 2 viết xong code tải CIFAR-10-C và $32$ phép tăng cường ảnh; Thành viên 3 chạy xong baseline *Source-only* và *BN-Adapt* có log lên W&B. |
| **Tuần 3** | **Tự cài đặt thuật toán CoTTA** | Thành viên 1 hoàn thiện class `CoTTA` (EMA + Augmentation Averaging + Stochastic Restoration) từ đầu bằng PyTorch; ghép nối với pipeline dữ liệu của Thành viên 2. |
| **Tuần 4** | **Chạy Tái lập Thực nghiệm (Reproduction)** | Chạy thực nghiệm tái lập CoTTA trên CIFAR-10-C (và CIFAR-100-C); tinh chỉnh siêu tham số đúng chuẩn bài báo ($\alpha=0.999, p=0.01, p_{\text{th}}=0.92$); lưu toàn bộ runs trên W&B. |
| **Tuần 5** | **Phân tích Độ lệch & Vẽ Sơ đồ** | Đối chiếu số liệu tái lập với bảng kết quả trong bài báo gốc; chạy thử trên $3$ random seeds để đo độ lệch chuẩn; tự vẽ toàn bộ hình minh họa phương pháp. |
| **Tuần 6 – 7** | **Hoàn thiện & Nộp Slide Deck ($45\%$)** | Ghép nội dung đọc hiểu và kết quả tái lập vào Slide Deck; đính kèm link W&B vào từng bảng số liệu; rà soát quy chuẩn định dạng và nộp Slide Deck + Link Repo lên Moodle. |
| **Tuần 8** | **Code 3 Kịch bản Phản biện (Critique Code)** | Thành viên 1 code *Guided Restoration*; Thành viên 2 code bộ chia dữ liệu *Non-i.i.d. Dirichlet* và *Small Batch*; Thành viên 3 code bộ đo *FLOPs/Latency* theo $N$. |
| **Tuần 9 – 10** | **Chạy Thực nghiệm Phản biện (Empirics)** | Chạy toàn bộ 3 nhóm thí nghiệm chứng minh điểm yếu trên GPU, đồng bộ kết quả lên W&B và xuất các biểu đồ chất lượng cao (vector PDF/PNG độ phân giải cao). |
| **Tuần 11** | **Thiết kế & Nộp Poster ($50\%$)** | Đưa lý thuyết phản biện, công thức toán và 3 nhóm biểu đồ thực nghiệm vào 1 trang Poster duy nhất; dọn dẹp code, viết hướng dẫn chạy lại chi tiết trong `README.md` ($5\%$). |
| **Tuần 12** | **Tập duyệt & Bảo vệ tại Poster Workshop** | Cả 3 thành viên tập thuyết trình chéo phần của nhau trước Poster; chuẩn bị câu trả lời cho các câu hỏi xoáy về lý thuyết và thực nghiệm của giảng viên. |

---

## 7. CẤU TRÚC THƯ MỤC GITHUB REPOSITORY CHUẨN CHO NHÓM

Nhóm thống nhất tổ chức code theo cấu trúc dưới đây để đạt trọn vẹn $5\%$ điểm chất lượng quản lý kho mã nguồn:

```text
Machine-learning-CoTTA/
├── README.md                       # Giới thiệu dự án, hướng dẫn cài đặt, câu lệnh chạy & bảng link W&B runs
├── requirements.txt                # Các thư viện: torch, torchvision, robustbench, wandb, numpy, matplotlib
├── configs/
│   ├── cifar10c_cotta.yaml         # Siêu tham số chuẩn cho CIFAR-10-C (lr=0.001, alpha=0.999, p=0.01, p_th=0.92)
│   └── cifar100c_cotta.yaml        # Siêu tham số chuẩn cho CIFAR-100-C
├── src/
│   ├── __init__.py
│   ├── datasets.py                 # Tải CIFAR-10-C / CIFAR-100-C và tạo luồng Non-i.i.d. Dirichlet (Thành viên 2)
│   ├── augmentations.py            # Tổ hợp N=32 phép biến đổi tăng cường dữ liệu ảnh (Thành viên 2)
│   ├── cotta.py                    # Cài đặt thuật toán CoTTA & bản cải tiến Guided Restoration (Thành viên 1)
│   ├── baselines.py                # Cài đặt Source-only, BN-Adapt, TENT để đối chứng (Thành viên 3)
│   └── utils.py                    # Khởi tạo W&B logger, cố định random seed, tính Error Rate & đo thời gian
├── experiments/
│   ├── run_reproduction.py         # Script chạy tái lập kết quả bài báo gốc (Phục vụ Slide Deck 45%)
│   ├── critique_1_restoration.py   # Script chạy thí nghiệm Điểm yếu 1: Độ nhạy p & Guided Restoration (Thành viên 1)
│   ├── critique_2_non_iid_batch.py # Script chạy thí nghiệm Điểm yếu 2: Batch Size nhỏ & Non-i.i.d. (Thành viên 2)
│   └── critique_3_complexity.py    # Script chạy thí nghiệm Điểm yếu 3: Độ trễ theo N & Ngưỡng p_th (Thành viên 3)
└── notebooks/
    └── plot_poster_figures.ipynb   # Notebook tổng hợp dữ liệu từ W&B để vẽ biểu đồ đưa lên Slide và Poster
```

---

## 8. QUY TẮC LÀM VIỆC NHÓM TRÊN GIT & W&B HÀNG TUẦN

1. **Quy tắc Commit GitHub:**
   * Mỗi thành viên làm việc trên nhánh riêng (ví dụ: `feat/cotta-core`, `feat/data-stream`, `feat/baselines`) hoặc pull code mới nhất (`git pull origin main`) trước khi code.
   * Ghi rõ thông điệp commit (ví dụ: `Implement EMA teacher update and stochastic restoration in src/cotta.py`).
   * Mỗi tuần cả 3 thành viên đều phải có commit đẩy lên repo để lịch sử đóng góp (Contributors graph) hiển thị đồng đều.
2. **Quy tắc Lưu trữ Weights & Biases (W&B):**
   * Đặt tên rõ ràng cho từng lượt chạy trên W&B theo cú pháp:  
     * Giai đoạn tái lập: `repro_cifar10c_cotta_seed42`, `repro_cifar10c_bn_adapt`
     * Giai đoạn phản biện: `critique1_guided_restoration_p0.01`, `critique2_batchsize_4`, `critique3_aug_N8`
   * Lưu lại đường link URL của từng lượt chạy quan trọng vào bảng kết quả trong `README.md` để giảng viên có thể bấm vào kiểm tra bất cứ lúc nào.