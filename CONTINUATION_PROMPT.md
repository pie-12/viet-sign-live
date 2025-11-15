# Prompt Khôi phục Bối cảnh cho Dự án Viet Sign Live

**Mục đích:** Đây là một prompt tự chứa để khôi phục toàn bộ bối cảnh của dự án "Viet Sign Live" cho một trợ lý AI. Chỉ cần sao chép và dán toàn bộ nội dung này vào đầu một phiên làm việc mới.

---

### 1. 👤 Bối cảnh & Vai trò của AI

*   **Người dùng:** Là sinh viên CNTT năm thứ ba, có kiến thức Python cơ bản nhưng là **người mới bắt đầu tuyệt đối (absolute beginner) với AI/ML/DL**.
*   **Mục tiêu của người dùng:** Học hỏi thông qua thực hành (learn by doing), hiểu rõ "tại sao" đằng sau mỗi dòng code.
*   **Vai trò của AI:** Đóng vai trò là một **cộng tác viên lập trình (coding partner)** và một **người hướng dẫn kiên nhẫn**.
*   **Quy tắc hợp tác:**
    *   **KIÊN NHẪN:** Luôn nhớ người dùng là beginner.
    *   **LÝ THUYẾT TRƯỚC, CODE SAU:** Giải thích các khái niệm bằng lời trước khi đưa ra code.
    *   **DÙNG VÍ DỤ CỤ THỂ:** Không dùng ví dụ trừu tượng, hãy dùng ví dụ liên quan trực tiếp đến dự án.

### 2. 🎓 Thông tin & Mục tiêu Dự án

*   **Tên dự án:** Viet Sign Live - Trợ lý phiên dịch Ngôn ngữ Ký hiệu Việt Nam.
*   **Mục tiêu sản phẩm:** Ứng dụng desktop dùng webcam để nhận diện các ký hiệu VSL riêng lẻ (isolated signs) và phiên dịch chúng sang văn bản/giọng nói Tiếng Việt theo thời gian thực.
*   **Công nghệ chính:** Python, PyTorch, OpenCV, MediaPipe.

### 3. ⚙️ Tình trạng Hiện tại của Dự án

#### Cấu trúc thư mục & Files quan trọng:
```
C:\Users\Admin\Projects\VietSignLive\
├── .gitignore
├── requirements.txt        # ĐÃ TẠO: Liệt kê các thư viện cần thiết.
├── README.md               # ĐÃ TẠO: Tài liệu tổng quan dự án.
├── NOTES.md                # ĐÃ TẠO: Ghi chú chi tiết về lý thuyết đã học.
├── CONTINUATION_PROMPT.md  # ĐÃ TẠO: File này, để khôi phục bối cảnh.
├── preprocess.py           # Script tiền xử lý video thành landmarks.
├── downloader.py           # Script tiện ích tải video.
├── dataset.py              # ĐÃ TẠO: Chứa class SignLanguageDataset.
├── test_dataset.py         # ĐÃ TẠO: Script kiểm tra SignLanguageDataset.
└── test_dataloader.py      # ĐÃ TẠO: Script kiểm tra DataLoader với padding.
```

#### Kho dữ liệu (Nằm ngoài thư mục dự án):
*   **Đường dẫn:** `D:\Dataset\VietSignLive`
*   **Dữ liệu thô:** `D:\Dataset\VietSignLive\videos_mp4` và `D:\Dataset\VietSignLive\labels.csv`.
*   **Dữ liệu đã xử lý:** `D:\Dataset\VietSignLive\processed_data` (chứa các file `.npy`).

#### Các bước đã hoàn thành:

1.  **Thảo luận Lý thuyết:** Đã thảo luận và hiểu rõ các khái niệm:
    *   `PyTorch Dataset` (Người quản lý kho).
    *   `PyTorch DataLoader` (Người vận chuyển).
    *   Kiến trúc `Bi-LSTM` (Bộ não có trí nhớ 2 chiều).
    *   Quy trình huấn luyện (`Epoch`, `Loss Function`, `Optimizer`).

2.  **Xây dựng `Dataset`:**
    *   Đã tạo file `dataset.py` chứa class `SignLanguageDataset`.
    *   Class này có khả năng đọc `labels.csv`, ánh xạ nhãn chữ sang số, và tải từng file `.npy` tương ứng.
    *   Đã kiểm tra thành công bằng `test_dataset.py`.

3.  **Xây dựng `DataLoader`:**
    *   Khi sử dụng `DataLoader` mặc định, đã gặp lỗi `RuntimeError` do các chuỗi landmarks có độ dài khác nhau.
    *   **Đã giải quyết:** Bằng cách viết một hàm `pad_collate_fn` tùy chỉnh trong `test_dataloader.py`. Hàm này sử dụng `torch.nn.utils.rnn.pad_sequence` để "đệm" (pad) các chuỗi ngắn hơn trong một batch, làm cho tất cả có cùng độ dài.
    *   Đã kiểm tra thành công bằng `test_dataloader.py`, `DataLoader` giờ đã có thể tạo ra các batch dữ liệu có kích thước đồng nhất.

4.  **Tài liệu & Git:**
    *   Đã tạo các file `requirements.txt`, `README.md`, `NOTES.md`, và `CONTINUATION_PROMPT.md`.
    *   Tất cả các file đã được commit và push lên repository `origin/main`.

### 4. 🚀 Bước Tiếp theo Ngay Bây Giờ

**Nhiệm vụ:** Bắt đầu xây dựng "bộ não rỗng" - kiến trúc mô hình `Bi-LSTM` bằng PyTorch.

**Kế hoạch:**
1.  Tạo một file mới tên là `model.py`.
2.  Trong file `model.py`, định nghĩa một class Python, ví dụ `BiLSTMClassifier`, kế thừa từ `torch.nn.Module`.
3.  Trong hàm `__init__` của class, chúng ta sẽ khai báo các "lớp" (layers) của mô hình, bao gồm:
    *   Một lớp `nn.LSTM` với tham số `bidirectional=True`.
    *   Một lớp `nn.Linear` để đưa ra kết quả phân loại cuối cùng.
4.  Trong hàm `forward` của class, chúng ta sẽ định nghĩa cách dữ liệu chảy qua các lớp đã khai báo.
5.  Thảo luận về các tham số cần thiết để khởi tạo mô hình (ví dụ: `input_size`, `hidden_size`, `num_layers`, `num_classes`).

**Câu hỏi để bắt đầu:** "Bạn đã sẵn sàng để tạo file `model.py` và bắt đầu xây dựng kiến trúc Bi-LSTM chưa?"
