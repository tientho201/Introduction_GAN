## Giới thiệu về GAN và Dataset

### Mục tiêu
- **Giới thiệu GAN**: Khái niệm, kiến trúc, huấn luyện, ứng dụng, hạn chế.
- **Giới thiệu Dataset**: Nguồn dữ liệu, cấu trúc, tiền xử lý, tiêu chí đánh giá.

---

## 1) Generative Adversarial Networks (GAN) là gì?

- **Khái niệm**: GAN gồm hai mạng nơ-ron đối kháng làm việc đồng thời:
  - **Generator (G)**: Nhận nhiễu ngẫu nhiên và sinh ra dữ liệu giả.
  - **Discriminator (D)**: Phân biệt dữ liệu thật và dữ liệu giả do G tạo.
- **Ý tưởng cốt lõi**: G cố gắng “lừa” D, trong khi D cố gắng phát hiện mẫu giả. Quá trình cạnh tranh này giúp G sinh dữ liệu ngày càng giống dữ liệu thật.

### Kiến trúc tổng quan
- **Input của G**: Vector nhiễu (Gaussian/Uniform).
- **Output của G**: Mẫu dữ liệu tổng hợp (ảnh/âm thanh/chuỗi…).
- **Input của D**: Mẫu thật hoặc giả.
- **Output của D**: Xác suất mẫu là thật.
- **Hàm mất mát**: D tối đa khả năng phân biệt đúng; G tối ưu để D dự đoán “thật” cho mẫu giả.

### Quy trình huấn luyện (vòng lặp đối kháng)
1. Lấy minibatch dữ liệu thật từ dataset.
2. Lấy nhiễu để G tạo dữ liệu giả.
3. Huấn luyện D trên dữ liệu thật và giả.
4. Cố định D, huấn luyện G để cải thiện chất lượng mẫu sinh.
5. Lặp lại cho đến khi hội tụ hoặc đạt chất lượng mong muốn.

### Biến thể GAN phổ biến
- **DCGAN**: Sử dụng CNN sâu, phù hợp dữ liệu ảnh.
- **WGAN / WGAN-GP**: Dùng khoảng cách Wasserstein, huấn luyện ổn định hơn, giảm mode collapse.
- **cGAN**: Điều kiện hóa theo nhãn để sinh dữ liệu theo lớp mong muốn.
- **StyleGAN**: Sinh ảnh chất lượng cao, kiểm soát phong cách tốt.

### Ứng dụng
- **Tạo ảnh tổng hợp**: Khuôn mặt, phong cảnh, sản phẩm.
- **Tăng cường dữ liệu** cho các nhiệm vụ phân loại/phân đoạn.
- **Siêu phân giải (Super-Resolution)**: Nâng độ phân giải ảnh.
- **Chuyển phong cách** và **dịch miền** (image-to-image translation).
- **Phát hiện bất thường**: Học phân phối bình thường để nhận diện outlier.

### Thách thức và hạn chế
- **Mode collapse**: G sinh ít dạng mẫu.
- **Bất ổn khi huấn luyện**: Nhạy với kiến trúc, siêu tham số, chuẩn hóa.
- **Đánh giá chất lượng**: Cần thước đo phù hợp (ví dụ FID) thay vì chỉ đánh giá bằng mắt.

---

## 2) Giới thiệu Dataset

Điều chỉnh nội dung phần này theo dataset được dùng trong `GAN.ipynb`. Nếu dùng bộ dữ liệu chuẩn (MNIST, CIFAR-10, CelebA…), cập nhật tên, nguồn và giấy phép tương ứng.

### Tổng quan
- **Tên dataset**: (Ví dụ) MNIST / CIFAR-10 / CelebA / Custom
- **Loại dữ liệu**: Ảnh xám 28×28 / Ảnh RGB 32×32 / Ảnh khuôn mặt 128×128 / Khác
- **Kích thước**: Số mẫu huấn luyện và kiểm định (nếu có)
- **Mục tiêu**: Huấn luyện GAN sinh dữ liệu cùng phân phối với dữ liệu thật

### Nguồn và giấy phép
- **Nguồn tải**: (Ví dụ) `https://www.kaggle.com/`, tài liệu `torchvision.datasets`, hoặc trang chủ dataset
- **Giấy phép**: Nêu rõ điều khoản (ví dụ: CC BY-NC 4.0, MIT, hoặc giấy phép riêng)


### Tiền xử lý dữ liệu
- **Chuẩn hóa (Normalization)**: Đưa pixel về [-1, 1] hoặc [0, 1] tùy kiến trúc
- **Resize/Crop**: Đồng nhất kích thước (ví dụ 64×64 cho DCGAN)
- **Augmentation (tùy chọn)**: Lật, xoay, jitter màu để tăng đa dạng
- **Tách tập**: Tách train/val/test nếu cần đánh giá định lượng

### Tiêu chí đánh giá chất lượng sinh
- **Định lượng**:
  - FID (Fréchet Inception Distance): Càng thấp càng tốt
  - IS (Inception Score): Đánh giá tính đa dạng và phân biệt lớp
- **Định tính**:
  - Quan sát mức độ sắc nét, artefact
  - Đa dạng mẫu, tránh lặp lại

### Lưu trữ và tái lập
- Ghi lại seed ngẫu nhiên
- Lưu siêu tham số: batch size, learning rate, số epoch, kiến trúc G/D, hàm mất mát
- Ghi rõ phiên bản/nguồn dataset để tái lập

---

## 3) Cách sử dụng dự án (không kèm mã)

- Mở notebook `GAN.ipynb`.
- Chạy lần lượt các cell để:
  - Tải/chuẩn bị dataset theo cấu trúc đã nêu
  - Cấu hình kiến trúc G/D và siêu tham số
  - Huấn luyện mô hình và lưu ảnh sinh vào `outputs/gan_samples`
  - Theo dõi log tổn thất của G/D theo epoch để phát hiện mất cân bằng
- Kiểm tra ảnh sinh theo từng mốc epoch để chọn mô hình tốt nhất.

---

## 4) Ghi chú thực hành tốt

- **Ổn định huấn luyện**:
  - Dùng chuẩn hóa phù hợp (BatchNorm trong G, SpectralNorm trong D nếu cần)
  - Khởi tạo trọng số hợp lý (Normal với độ lệch chuẩn nhỏ)
  - Tỷ lệ cập nhật D:G có thể điều chỉnh (ví dụ 1:1 hoặc 5:1)
- **Siêu tham số gợi ý**:
  - Optimizer: Adam với beta1 ≈ 0.5 giúp ổn định cho DCGAN
  - Learning rate vừa phải để tránh dao động
- **Giảm mode collapse**:
  - Quan sát đa dạng ảnh sinh, thử giảm LR hoặc dùng WGAN-GP
  - Có thể dùng label smoothing nhẹ cho D (tùy bài toán)

---

## 5) Tài liệu tham khảo

- Goodfellow et al., 2014 — Generative Adversarial Nets
- Radford et al., 2016 — DCGAN
- Arjovsky et al., 2017 — Wasserstein GAN
- Gulrajani et al., 2017 — WGAN-GP
- Karras et al., 2019–2021 — StyleGAN series

Nếu bạn cung cấp tên dataset cụ thể dùng trong `GAN.ipynb`, tôi sẽ cập nhật phần Dataset với nguồn, giấy phép, kích thước và hướng dẫn tiền xử lý chi tiết hơn.


