# Ghi chú Dự án Viet Sign Live

Đây là file ghi chú tổng hợp lại các khái niệm, lý thuyết và các bước đã thực hiện trong dự án "Viet Sign Live".

## 1. 🎯 Tầm nhìn & Mục tiêu

Mục tiêu cuối cùng là xây dựng một ứng dụng desktop có khả năng phiên dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) sang văn bản và giọng nói theo thời gian thực.

Dự án áp dụng kiến trúc 2-Mô-hình:
1.  **Mô hình 1 (Trích xuất Đặc trưng):** Dùng `MediaPipe` có sẵn để biến đổi mỗi frame video thành một bộ tọa độ "bộ xương" (landmarks).
2.  **Mô hình 2 (Phân loại Ký hiệu):** Tự xây dựng và huấn luyện một mô hình Deep Learning (bắt đầu với `Bi-LSTM`) để đọc chuỗi landmarks và đoán ra đó là ký hiệu gì.

---

## 2. 🗂️ Quy trình Xử lý Dữ liệu (Data Pipeline)

Đây là các bước để chuẩn bị dữ liệu từ video thô cho đến khi sẵn sàng cho mô hình PyTorch.

### Bước 1: Tiền xử lý - "Thu hoạch Bộ xương"

-   **Mục tiêu:** Chuyển đổi hàng nghìn video `.mp4` thành dữ liệu có cấu trúc mà mô hình có thể học được.
-   **Công cụ:** Script `preprocess.py` (sử dụng thư viện `mediapipe`).
-   **Đầu vào:** Thư mục `videos_mp4` chứa các file video gốc.
-   **Đầu ra:**
    -   Thư mục `processed_data`: Chứa hàng nghìn file `.npy`. Mỗi file `.npy` là một mảng NumPy, lưu trữ chuỗi landmarks của một video.
    -   File `labels.csv`: File "đáp án", ánh xạ mỗi file `.npy` với nhãn (tên ký hiệu) tương ứng.

### Bước 2: Tải dữ liệu cho PyTorch

Đây là phần chúng ta đã làm cùng nhau để đưa dữ liệu `.npy` vào PyTorch.

#### Khái niệm 1: `PyTorch Dataset` - Người Quản lý Kho

-   **Nhiệm vụ:** Định nghĩa cách PyTorch truy cập vào **từng mẫu dữ liệu riêng lẻ**. Nó không tải toàn bộ dữ liệu vào bộ nhớ mà chỉ tải khi được yêu cầu.
-   **Triển khai của chúng ta:** Class `SignLanguageDataset` trong file `dataset.py`.
-   **Cách hoạt động:**
    1.  Khi khởi tạo, nó đọc `labels.csv` để biết danh sách tất cả các mẫu.
    2.  Nó tạo ra một từ điển để **ánh xạ nhãn dạng chữ (ví dụ: "XIN CHAO") sang nhãn dạng số (ví dụ: 0)**, vì mô hình học máy làm việc hiệu quả hơn với số.
    3.  Khi được `DataLoader` yêu cầu một mẫu tại `index` cụ thể, hàm `__getitem__` sẽ được gọi.
    4.  Hàm `__getitem__` sẽ đọc file `.npy` tương ứng, lấy nhãn, và trả về một cặp dữ liệu đã được chuyển đổi thành `PyTorch Tensor`.

#### Khái niệm 2: `PyTorch DataLoader` - Người Vận chuyển Hàng hóa

-   **Nhiệm vụ:** Lấy dữ liệu từ `Dataset` và gom chúng lại thành từng **lô (batch)** để cung cấp cho mô hình. Việc xử lý theo batch giúp quá trình huấn luyện hiệu quả hơn.
-   **Các tham số quan trọng:**
    -   `batch_size`: Số lượng mẫu trong một lô (ví dụ: 32).
    -   `shuffle=True`: Xáo trộn dữ liệu ở mỗi `epoch` để mô hình không học vẹt thứ tự và có thể tổng quát hóa tốt hơn.

#### Vấn đề & Giải pháp: Dữ liệu có độ dài khác nhau

-   **Vấn đề gặp phải:** Các video có độ dài khác nhau, dẫn đến các chuỗi landmarks có số frame khác nhau. `DataLoader` mặc định không thể gom các tensor có kích thước không đồng nhất vào cùng một batch. Điều này gây ra lỗi `RuntimeError: stack expects each tensor to be equal size`.
-   **Giải pháp:** Kỹ thuật **Padding (Đệm)**.
-   **Triển khai của chúng ta:**
    1.  Chúng ta đã viết một hàm tùy chỉnh gọi là `pad_collate_fn`.
    2.  Hàm này được truyền vào `DataLoader` qua tham số `collate_fn`.
    3.  Nó sử dụng hàm `pad_sequence` của PyTorch để tự động tìm chuỗi dài nhất trong một batch và "đệm" các chuỗi ngắn hơn bằng số 0 cho đến khi tất cả có cùng độ dài.
    4.  Kết quả là các batch dữ liệu có kích thước đồng nhất và sẵn sàng cho mô hình.

---

## 3. 🧠 Lý thuyết Mô hình & Huấn luyện

Đây là các khái niệm lý thuyết chúng ta đã thảo luận để chuẩn bị cho việc xây dựng và huấn luyện mô hình.

### Kiến trúc Mô hình: `Bi-LSTM`

-   **Tại sao cần kiến trúc đặc biệt?** Dữ liệu của chúng ta là dạng chuỗi (sequence), nơi thứ tự các frame là rất quan trọng. Các mạng nơ-ron thông thường không có "trí nhớ" về các frame trước đó.
-   **LSTM (Long Short-Term Memory):** Một loại mạng nơ-ron có "trí nhớ". Nó có các "cổng" (gates) thông minh để quyết định thông tin nào cần **ghi nhớ**, thông tin nào cần **quên đi** khi xử lý một chuỗi. Điều này giúp nó hiểu được bối cảnh của các chuyển động.
-   **Bi-LSTM (Bi-directional LSTM):** Là sự kết hợp của hai mạng LSTM: một mạng chạy **xuôi** (từ đầu đến cuối chuỗi) và một mạng chạy **ngược** (từ cuối về đầu). Điều này cho phép mô hình có được bối cảnh từ cả quá khứ và tương lai, giúp nó đưa ra dự đoán chính xác hơn, đặc biệt hữu ích cho ngôn ngữ ký hiệu.

### Quá trình Huấn luyện (Training Loop)

-   **`Epoch`:** Một `epoch` là khi mô hình đã "nhìn" và học hỏi từ **toàn bộ** dữ liệu huấn luyện một lần. Chúng ta thường huấn luyện trong nhiều `epoch`.
-   **`Loss Function` (Hàm mất mát):** Là "thước đo" mức độ sai của dự đoán của mô hình so với đáp án đúng. Mục tiêu của quá trình huấn luyện là làm cho giá trị `loss` này **càng nhỏ càng tốt**.
-   **`Optimizer` (Bộ tối ưu hóa):** Là cơ chế (ví dụ: Adam, SGD) giúp **cập nhật** các tham số bên trong của mô hình dựa trên giá trị `loss`. Nó điều chỉnh mô hình theo hướng làm cho dự đoán lần sau tốt hơn.

---

## 4. 🚀 Các bước tiếp theo

-   **Xây dựng kiến trúc mô hình Bi-LSTM** bằng các lớp (layers) của PyTorch.
-   Viết vòng lặp huấn luyện (training loop) để đưa dữ liệu từ `DataLoader` vào mô hình, tính toán `loss`, và dùng `optimizer` để cập nhật mô hình.
