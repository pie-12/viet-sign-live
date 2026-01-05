# 🤟 VietSignLive - Vietnamese Sign Language Recognition

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green.svg)

**VietSignLive** là hệ thống nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL) thời gian thực, sử dụng Deep Learning để chuyển đổi cử chỉ tay và cơ thể thành văn bản. Dự án được xây dựng với mục tiêu hỗ trợ giao tiếp cho cộng đồng người khiếm thính tại Việt Nam.

## 🌟 Điểm Nổi Bật (Key Features)

- **Quy mô dữ liệu lớn:** Hỗ trợ nhận diện **2764 từ vựng** và bảng chữ cái, được thu thập từ nguồn chính thống (Bộ Giáo dục & Đào tạo).
- **Mô hình mạnh mẽ:** Kiến trúc **Deep Bidirectional LSTM (3 lớp)** giúp nắm bắt tốt các chuỗi hành động theo thời gian.
- **Kỹ thuật Augmentation nâng cao:** Áp dụng thuật toán **Inverse Kinematics (IK) 2D** để mô phỏng sự thay đổi khoảng cách tay và cấu trúc cơ thể, kết hợp với các phép biến đổi hình học (Scale, Rotate, Time Stretch).
- **Giao diện hiện đại (HUD Style):** Ứng dụng Streamlit với thiết kế High-Contrast, hỗ trợ chế độ Webcam thời gian thực và upload video.
- **Xử lý hiệu năng cao:** Tối ưu hóa MediaPipe Holistic để trích xuất 201 điểm đặc trưng (Landmarks) trên mỗi khung hình.

## 🛠️ Kiến Trúc Hệ Thống

### 1. Data Pipeline
- **Nguồn:** Crawl tự động từ `qipedc.moet.gov.vn` sử dụng Selenium.
- **Preprocessing:**
    - Trích xuất 201 keypoints (Pose + Left Hand + Right Hand) bằng MediaPipe.
    - Chuẩn hóa độ dài chuỗi về **60 frames** (sử dụng nội suy Cubic).
- **Augmentation:**
    - `solve_2_link_ik_2d_v2`: Giải bài toán động học ngược để thay đổi vị trí tay tự nhiên.
    - `time_stretch`: Co giãn thời gian để giả lập tốc độ ký hiệu nhanh/chậm.

### 2. Model Architecture
Mô hình được huấn luyện trên TensorFlow/Keras:
- **Input:** `(60 frames, 201 features)`
- **Hidden Layers:**
    - 3x Bidirectional LSTM Layers (256 units/layer)
    - Batch Normalization & Dropout (0.3 - 0.5) để chống Overfitting.
- **Output:** Dense Layer (2764 units) với hàm kích hoạt Softmax.

## 🚀 Cài Đặt & Sử Dụng

### Yêu cầu hệ thống
- Python 3.8+
- Webcam (cho tính năng nhận diện trực tiếp)

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
Khởi động giao diện web local:
```bash
streamlit run main.py
```
Sau khi chạy, truy cập vào đường dẫn hiển thị trên terminal (thường là `http://localhost:8501`).

### 3. Huấn luyện lại (Optional)
Nếu bạn muốn tự huấn luyện lại mô hình từ dữ liệu gốc:
1. Thu thập dữ liệu: `python download_data.py`
2. Xử lý và Augment dữ liệu: `python create_data_augment.py`
3. Huấn luyện: Mở và chạy notebook `trainning.ipynb`

## 📂 Cấu Trúc Dự Án

```
VietSignLive/
├── main.py                 # Giao diện chính (Streamlit App)
├── download_data.py        # Script crawl dữ liệu từ Bộ GD&ĐT
├── augment_function.py     # Thư viện các hàm Data Augmentation (IK, Rotate...)
├── create_data_augment.py  # Script tạo dataset huấn luyện
├── trainning.ipynb         # Notebook huấn luyện model
├── requirements.txt        # Danh sách thư viện
├── Dataset/                # Chứa video gốc và file CSV nhãn
├── Models/
│   └── checkpoints/
│       └── final_model.keras  # File model đã huấn luyện
└── Logs/
    └── label_map.json      # Mapping giữa ID và nhãn từ vựng
```

## 👨‍💻 Tác Giả

**Tung Lam Nguyen**
- Student ID: 23IT138
- Project: Vietnamese Sign Language Recognition System

---
*Dự án phục vụ mục đích học tập và nghiên cứu.*
