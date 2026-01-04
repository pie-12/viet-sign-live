# Viet Sign Live - Trợ lý phiên dịch Ngôn ngữ Ký hiệu Việt Nam

Dự án "Viet Sign Live" là một ứng dụng desktop nhằm mục đích phiên dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) sang văn bản và giọng nói Tiếng Việt theo thời gian thực. Dự án được phát triển như một phần của Đồ án cơ sở 4.

## 🎯 Mục tiêu

-   **Nhận diện thời gian thực:** Sử dụng webcam để nhận diện các ký hiệu VSL riêng lẻ (isolated signs).
-   **Phiên dịch đa phương thức:** Chuyển đổi ký hiệu được nhận diện thành văn bản và phát ra giọng nói tương ứng.
-   **Xử lý trên thiết bị:** Ưu tiên xử lý on-device để đảm bảo quyền riêng tư và tốc độ, chỉ sử dụng tọa độ landmarks thay vì hình ảnh/video gốc.
-   **Học hỏi & Nghiên cứu:** Dự án được xây dựng với mục tiêu chính là học hỏi và áp dụng các kỹ thuật Deep Learning vào bài toán nhận diện hành động.

## 🛠️ Công nghệ (Tech Stack)

-   **Ngôn ngữ:** Python 3.11
-   **Trích xuất Đặc trưng:** MediaPipe (Holistic)
-   **Deep Learning Framework:** PyTorch
-   **Xử lý Video/Ảnh:** OpenCV
-   **Thao tác dữ liệu:** Pandas, NumPy
-   **Giao diện (dự kiến):** Gradio / PyQt
-   **Tối ưu hóa (dự kiến):** ONNX Runtime

## 🗂️ Cấu trúc Dự án

```
VietSignLive/
│
├── .venv/                  # Môi trường ảo Python
├── data/                   # (Thư mục dữ liệu không nằm trong repo)
│   ├── videos_mp4/         # Dữ liệu video thô
│   └── processed_data/     # Dữ liệu landmarks đã xử lý (.npy)
│
├── .gitignore              # Các file/thư mục được Git bỏ qua
├── requirements.txt        # Danh sách các thư viện cần thiết
├── preprocess.py           # Script tiền xử lý, chuyển video thành landmarks
├── dataset.py              # PyTorch Dataset để tải dữ liệu landmarks
├── test_dataset.py         # Script để kiểm tra Dataset
├── downloader.py           # Script tiện ích để tải video
└── README.md               # File này
```

## 🚀 Cài đặt & Thiết lập

1.  **Clone repository:**
    ```bash
    git clone https://github.com/pie-12/viet-sign-live.git
    cd viet-sign-live
    ```

2.  **Tạo và kích hoạt môi trường ảo:**
    ```bash
    # Tạo môi trường ảo
    python -m venv .venv

    # Kích hoạt môi trường ảo (Windows)
    .\.venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Cách sử dụng

1.  **Tiền xử lý dữ liệu (Tùy chọn, nếu bạn có dữ liệu video thô):**
    Chạy script `preprocess.py` để chuyển đổi các video `.mp4` thành các file landmarks `.npy`. Bạn cần cấu hình đường dẫn đến thư mục video và thư mục đầu ra trong file.

2.  **Kiểm tra `Dataset`:**
    Để đảm bảo `PyTorch Dataset` có thể đọc dữ liệu `.npy` và `labels.csv` một cách chính xác, hãy chạy script kiểm tra:
    ```bash
    python test_dataset.py
    ```
    *Lưu ý: Bạn cần cập nhật đường dẫn đến thư mục dữ liệu trong file `test_dataset.py` cho phù hợp với máy của bạn.*
