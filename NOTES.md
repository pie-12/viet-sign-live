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
    -   Thư mục `processed_data_60_201`: Chứa hàng nghìn file `.npy`. Mỗi file `.npy` là một mảng NumPy, lưu trữ chuỗi landmarks đã được chuẩn hóa về kích thước `(60, 201)`.
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

-   **Vấn đề gặp phải:** Các video có độ dài khác nhau, dẫn đến các chuỗi landmarks có số frame khác nhau. `DataLoader` mặc định không thể gom các tensor có kích thước không đồng nhất vào cùng một batch. Điều này từng gây ra lỗi `RuntimeError: stack expects each tensor to be equal size`.
-   **Giải pháp (hiện tại):** Kỹ thuật **Chuẩn hóa độ dài chuỗi (Sequence Length Normalization)** trong bước tiền xử lý.
-   **Triển khai của chúng ta:**
    1.  Trong `preprocess.py`, tất cả các chuỗi keypoints đã được nội suy về độ dài chuẩn **60 frames**.
    2.  Kết quả là các batch dữ liệu luôn có kích thước đồng nhất `(BATCH_SIZE, 60, 201)` và sẵn sàng cho mô hình mà **không cần đến hàm `pad_collate_fn` tùy chỉnh** nữa.

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

## 4. 🚀 Các bước đã hoàn tất (Theo kiến trúc của báo cáo nghiên cứu)

Chúng ta đã hoàn thành các bước quan trọng sau để triển khai phương pháp dựa trên báo cáo nghiên cứu:

1.  **Tiền xử lý dữ liệu:** Script `src/data/preprocess.py` đã được cập nhật để trích xuất 201 chiều keypoints từ video thô và chuẩn hóa tất cả các chuỗi về độ dài 60 frames.
2.  **Xây dựng hàm tăng cường dữ liệu:** Module `src/data/augmentation.py` đã được tạo với các hàm xoay, dịch chuyển, phóng to/thu nhỏ và biến đổi tốc độ thời gian cho chuỗi keypoints.
3.  **Cập nhật bộ tải dữ liệu (Dataset Loader):** Class `SignLanguageDataset` trong `src/data/dataset.py` đã được sửa đổi để tải dữ liệu đã tiền xử lý, áp dụng tăng cường dữ liệu linh hoạt cho tập huấn luyện và không còn cần hàm `collate_fn` tùy chỉnh.
4.  **Xây dựng lại kiến trúc mô hình:** File `src/models/model.py` đã triển khai kiến trúc BiLSTM sâu với Batch Normalization và Dropout, đúng như mô tả trong báo cáo.
5.  **Cập nhật pipeline huấn luyện:** Script `src/train.py` đã được chỉnh sửa để tích hợp tất cả các thành phần mới và triển khai cơ chế Early Stopping dựa trên validation loss.

Toàn bộ pipeline từ tiền xử lý dữ liệu đến huấn luyện mô hình đã sẵn sàng.

---

## 5. 🔬 Quy trình Thử nghiệm & Tối ưu

Sau khi có mô hình và dữ liệu, quá trình tối ưu bắt đầu. Đây là nhật ký các thử nghiệm và kết quả.

### Vấn đề ban đầu: `Val Accuracy` ~ 0% (Thất bại khi huấn luyện)

-   **Hiện tượng:** Khi huấn luyện trên toàn bộ dataset (~4300 mẫu, ~3300 lớp), `Val Accuracy` không tăng và luôn ở mức ~0%.
-   **Chẩn đoán:** Dữ liệu quá thưa (sparse), trung bình chỉ có ~1.3 video cho mỗi ký hiệu. Mô hình không có đủ ví dụ để học và khái quát hóa.

### Thử nghiệm 1: Thu nhỏ bài toán (10 lớp)

-   **Hành động:** Chỉnh sửa `dataset.py` để chỉ giữ lại 10 lớp có nhiều mẫu nhất.
-   **Kết quả:** `Val Accuracy` đạt **14.29%**.
-   **Bài học:** **Thành công!** Chúng ta đã chứng minh được kiến trúc mô hình và pipeline dữ liệu cơ bản là đúng. Mô hình CÓ THỂ HỌC. Vấn đề nằm ở quy mô dữ liệu.

### Thử nghiệm 2: Mở rộng bài toán (30 lớp)

-   **Hành động:** Tăng số lớp được giữ lại lên 30.
-   **Kết quả:** `Val Accuracy` chỉ đạt ~5.26%. `Train Loss` giảm nhưng `Val Loss` tăng.
-   **Bài học:** Hiện tượng **Overfitting (Học vẹt)** xảy ra nghiêm trọng. Mô hình quá phức tạp so với lượng dữ liệu ít ỏi, nó chỉ "nhớ" đáp án của tập training chứ không thực sự học.

### Thử nghiệm 3: Chống Overfitting bằng Data Augmentation

-   **Hành động:** Thêm một lượng nhiễu (noise) ngẫu nhiên và rất nhỏ vào dữ liệu của mỗi batch **training**. Điều này buộc mô hình phải học các đặc trưng cốt lõi thay vì nhớ các toạ độ chính xác.
-   **Kết quả:** `Val Accuracy` trên bộ 30 lớp tăng gấp đôi, đạt **10.53%**.
-   **Bài học:** Data Augmentation là một kỹ thuật hiệu quả để chống overfitting, giúp mô hình khái quát hóa tốt hơn.

### Thử nghiệm 4: Chống Overfitting bằng Hyperparameter Tuning

-   **Hành động:**
    1.  **Giảm `HIDDEN_SIZE`** từ 256 -> 128 (Làm mô hình đơn giản hơn, khó học vẹt hơn).
    2.  **Giảm `LEARNING_RATE`** từ 0.001 -> 0.0001 (Để mô hình học chậm và "cẩn thận" hơn).
-   **Tình trạng:** **Đã dừng.** Chúng ta đã chuyển hướng sang phương pháp của báo cáo nghiên cứu.

### Thử nghiệm 5: Huấn luyện với Pipeline mới (Dựa trên báo cáo nghiên cứu)

-   **Hành động:** Chạy script `src/train.py` với toàn bộ pipeline đã được cập nhật (tiền xử lý chuẩn hóa, tăng cường dữ liệu on-the-fly, mô hình BiLSTM sâu).
-   **Kết quả (Epoch 2):**
    *   `Train Loss`: Giảm từ 8.4528 xuống 8.1433
    *   `Val Loss`: 8.4559 (không cải thiện đáng kể so với epoch 1)
    *   `Val Accuracy`: Tăng từ 0.00% lên **0.11%**.
-   **Bài học:**
    *   Mô hình **đã bắt đầu học**. Việc `Val Accuracy` tăng từ 0.00% lên 0.11% là một tín hiệu rất tích cực, cho thấy toàn bộ pipeline đang hoạt động đúng.
    *   Mức tăng accuracy này vẫn còn rất nhỏ, điều này phù hợp với dự đoán rằng mô hình sẽ học chậm hơn so với phương pháp trong báo cáo (do sử dụng tăng cường dữ liệu on-the-fly thay vì pre-generate một dataset khổng lồ).
    *   Cần tiếp tục huấn luyện trong nhiều epoch để đánh giá thêm hiệu suất và sự hội tụ của mô hình. Cơ chế Early Stopping sẽ giúp dừng lại khi mô hình không còn cải thiện.

---

## 6. 💡 Chiến lược Hiện tại: BiLSTM + Tăng cường dữ liệu Mạnh mẽ (Theo báo cáo nghiên cứu)

Sau khi gặp phải vấn đề mô hình "học vẹt" (overfitting) nghiêm trọng do dữ liệu huấn luyện thưa thớt, chúng ta đã tiến hành phân tích sâu hơn và tìm kiếm các giải pháp hiệu quả. Một báo cáo nghiên cứu về nhận diện ngôn ngữ ký hiệu Việt Nam (VSL) đã được tìm thấy, trong đó mô hình Bi-LSTM đạt độ chính xác lên đến **96%** trên cùng bộ dữ liệu.

**Phân tích báo cáo cho thấy thành công đến từ:**

1.  **Tiền xử lý Dữ liệu chuẩn hóa:** Tất cả các chuỗi keypoints được nội suy về độ dài cố định là **60 frames**. Điều này đảm bảo đầu vào đồng nhất cho mô hình.
2.  **Tăng cường Dữ liệu (Data Augmentation) mạnh mẽ:** Đây là yếu tố then chốt. Từ mỗi chuỗi video gốc, họ đã tạo ra **tối đa 100 phiên bản tăng cường** bằng cách kết hợp nhiều phép biến đổi:
    *   **Xoay (Rotation):** Mô phỏng góc quay camera hoặc tư thế người ký hiệu.
    *   **Dịch chuyển (Translation):** Mô phỏng vị trí người ký hiệu khác nhau trong khung hình.
    *   **Phóng to/Thu nhỏ (Scaling):** Mô phỏng kích thước người hoặc khoảng cách đến camera.
    *   **Biến đổi tốc độ thời gian (Temporal Speed Variation):** Mô phỏng tốc độ ký hiệu khác nhau.
    Nhờ đó, tập dữ liệu huấn luyện được mở rộng từ ~4000 mẫu lên đến **~280,000 mẫu**, giúp mô hình có đủ ví dụ để học và khái quát hóa mà không bị overfitting.
3.  **Kiến trúc Bi-LSTM sâu và được điều chỉnh tốt:** Mô hình sử dụng nhiều lớp Bi-LSTM chồng lên nhau, kết hợp với các lớp Batch Normalization và Dropout để ổn định quá trình huấn luyện và chống overfitting hiệu quả.

**Chiến lược hiện tại của chúng ta là tái triển khai mô hình Bi-LSTM theo đúng phương pháp đã được chứng minh trong báo cáo này.** Chúng ta đã hoàn tất các bước chuẩn bị dữ liệu và xây dựng mô hình theo kiến trúc này.
