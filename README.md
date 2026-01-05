# 🤟 VietSignLive - Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu Việt Nam

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00CC00?logo=google&logoColor=white)

**VietSignLive** là giải pháp ứng dụng Trí tuệ nhân tạo (AI) để nhận diện và phiên dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) thành văn bản theo thời gian thực. Dự án được xây dựng với mục tiêu xóa bỏ rào cản giao tiếp cho cộng đồng người khiếm thính.

---

## 🌟 Điểm Nổi Bật (Highlights)

*   **Quy mô dữ liệu lớn:** Hỗ trợ nhận diện **2764 từ vựng**, bao gồm cả các từ địa phương (Bắc - Trung - Nam) và bảng chữ cái/số. Dữ liệu chuẩn được thu thập từ Từ điển của Bộ Giáo dục & Đào tạo.
*   **Mô hình Deep Bi-LSTM:** Sử dụng kiến trúc mạng 3 lớp Bidirectional LSTM (Long Short-Term Memory) mạnh mẽ để nắm bắt các đặc trưng chuỗi thời gian của hành động phức tạp.
*   **Siêu tăng cường dữ liệu (Advanced Augmentation):**
    *   **Inverse Kinematics (IK 2D):** Thuật toán mô phỏng sự thay đổi khoảng cách tay nhưng vẫn giữ cấu trúc xương tự nhiên.
    *   **Geometric Transforms:** Xoay, Tịnh tiến, Phóng to/Thu nhỏ (Scale) dựa trên trọng tâm cơ thể.
    *   **Time Stretching:** Giả lập tốc độ ký hiệu nhanh/chậm.
*   **Xử lý Real-time thông minh:**
    *   **Normalization:** Tự động chuẩn hóa dữ liệu đầu vào theo tỷ lệ cơ thể người dùng (bất kể đứng xa/gần).
    *   **Frame Skipping:** Kỹ thuật lấy mẫu thông minh để giảm nhiễu và đồng bộ FPS.
*   **Giao diện HUD Modern:** Thiết kế Futuristic, trực quan với các chỉ số tin cậy (Confidence) và hiệu ứng gương.

---

## 🛠️ Kiến Trúc Hệ Thống

### 1. Luồng xử lý (Pipeline)
1.  **Input:** Webcam Stream hoặc Video File.
2.  **Feature Extraction:** MediaPipe Holistic trích xuất **201 điểm (Keypoints)** bao gồm Tư thế (Pose) và hai bàn tay (Hands).
3.  **Preprocessing:**
    *   **Clip & Normalize:** Chặn giá trị nhiễu và chuẩn hóa toạ độ về `[0, 1]`.
    *   **Interpolation:** Thuật toán Cubic nắn chỉnh chuỗi frame về độ dài cố định **60 frames**.
4.  **Inference:** Model Bi-LSTM dự đoán nhãn từ vựng.
5.  **Output:** Hiển thị kết quả và biểu đồ xác suất lên giao diện Streamlit.

### 2. Cấu trúc Model (Model Architecture)
*   **Input Shape:** `(Batch, 60, 201)`
*   **Hidden Layers:**
    *   3x Bidirectional LSTM (256 units/layer)
    *   Batch Normalization & Dropout (0.3 - 0.5) để tối ưu hóa và chống Overfitting.
*   **Output Layer:** Dense 2764 (Softmax).

---

## 🚀 Cài Đặt & Hướng Dẫn Sử Dụng

### Yêu cầu hệ thống
*   Python 3.8 - 3.11
*   Webcam (để chạy chế độ Real-time)
*   RAM: 8GB+ (Khuyến nghị)

### Bước 1: Cài đặt môi trường
```bash
# Clone dự án
git clone https://github.com/your-repo/VietSignLive.git
cd VietSignLive

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### Bước 2: Chạy ứng dụng
Mở terminal tại thư mục dự án và chạy lệnh:
```bash
streamlit run main.py
```
Trình duyệt sẽ tự động mở địa chỉ `http://localhost:8501`.

### Bước 3: Sử dụng
*   **Tab Camera:** Chọn thời lượng ghi (ví dụ 3s), bấm nút **"KÍCH HOẠT CAMERA"**. Đợi đếm ngược và thực hiện ký hiệu.
*   **Tab Video:** Tải file MP4/AVI lên để nhận diện.

---

## 📂 Cấu Trúc Thư Mục

```
VietSignLive/
├── main.py                 # Giao diện chính & Logic Real-time (Streamlit)
├── trainning.ipynb         # Notebook xây dựng & huấn luyện Model
├── augment_function.py     # Thư viện thuật toán Tăng cường dữ liệu (IK, Scale...)
├── create_data_augment.py  # Script xử lý dữ liệu thô -> Training data (.npz)
├── download_data.py        # Script crawl dữ liệu từ web Bộ GD&ĐT
├── requirements.txt        # Danh sách thư viện Python
├── Dataset/                # Thư mục chứa dữ liệu video & nhãn
├── Models/
│   └── checkpoints/
│       └── final_model.keras  # File Model đã huấn luyện (Core AI)
└── Logs/
    └── label_map.json      # Mapping ID <-> Nhãn từ vựng
```

---

## 👨‍💻 Tác Giả & Liên Hệ

**Nguyễn Tùng Lâm (Tung Lam Nguyen)**
*   **Student ID:** 23IT138
*   **Dự án:** Đồ án Cơ sở / Nghiên cứu khoa học - Hệ thống nhận diện Ngôn ngữ Ký hiệu.

---
*Dự án được xây dựng với tâm huyết hỗ trợ cộng đồng. Mọi đóng góp và ý kiến phản hồi đều được trân trọng.*