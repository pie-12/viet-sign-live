# Viet Sign Live - Trợ lý phiên dịch Ngôn ngữ Ký hiệu Việt Nam

Dự án "Viet Sign Live" là một ứng dụng nhằm mục đích phiên dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) sang văn bản và giọng nói Tiếng Việt theo thời gian thực. Dự án được phát triển như một phần của Đồ án cơ sở 4.

## 🎯 Mục tiêu

-   **Nhận diện thời gian thực:** Sử dụng webcam để nhận diện các ký hiệu VSL riêng lẻ (isolated signs).
-   **Phiên dịch đa phương thức:** Chuyển đổi ký hiệu được nhận diện thành văn bản và phát ra giọng nói tương ứng.
-   **Xử lý trên thiết bị:** Ưu tiên xử lý on-device để đảm bảo quyền riêng tư và tốc độ, chỉ sử dụng tọa độ landmarks thay vì hình ảnh/video gốc.

## 🛠️ Công nghệ (Tech Stack)

-   **Ngôn ngữ:** Python
-   **Trích xuất Đặc trưng:** MediaPipe (Holistic)
-   **Deep Learning Framework:** TensorFlow / Keras
-   **Giao diện:** Streamlit
-   **Xử lý Video/Ảnh:** OpenCV
-   **Thao tác dữ liệu:** Pandas, NumPy

## 🗂️ Cấu trúc Dự án

```
VietSignLive/
├── main.py                 # Ứng dụng Streamlit (Giao diện chính)
├── download_data.py        # Script tải dữ liệu video từ từ điển VSL
├── create_data_augment.py  # Tiền xử lý và tăng cường dữ liệu (Data Augmentation)
├── augment_function.py     # Các hàm bổ trợ cho quá trình tăng cường dữ liệu
├── trainning.ipynb         # Notebook huấn luyện mô hình
├── requirements.txt        # Danh sách các thư viện cần thiết
├── Dataset/                # Thư mục chứa dữ liệu
│   └── Text/label.csv      # File nhãn dữ liệu
├── Models/                 # Thư mục lưu trữ mô hình
│   └── checkpoints/        # File final_model.keras
├── Logs/                   # Log và label map
└── old/                    # Các phiên bản code cũ (PyTorch)
```

## 🚀 Cài đặt & Thiết lập

1.  **Clone repository:**
    ```bash
    git clone https://github.com/pie-12/viet-sign-live.git
    cd viet-sign-live
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Cách sử dụng

### 1. Chạy ứng dụng (Giao diện Streamlit)
Để sử dụng mô hình đã huấn luyện sẵn với webcam hoặc file video:
```bash
streamlit run main.py
```

### 2. Huấn luyện mô hình từ đầu
Nếu bạn muốn tự xây dựng lại mô hình:
1.  **Tải dữ liệu:** `python download_data.py`
2.  **Tiền xử lý & Tăng cường:** `python create_data_augment.py`
3.  **Huấn luyện:** Mở và chạy toàn bộ các cell trong file `trainning.ipynb`.

---
*Lưu ý: Các phiên bản sử dụng PyTorch cũ đã được chuyển vào thư mục `old/`.*