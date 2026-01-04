# Báo cáo Dự án: Viet Sign Live
### Xây dựng hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam

---

## 1. Giới thiệu

### 🎯 Mục tiêu Dự án
Mục tiêu cuối cùng của dự án "Viet Sign Live" là xây dựng một ứng dụng có khả năng phiên dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) sang văn bản theo thời gian thực, nhằm hỗ trợ cộng đồng người khiếm thính trong giao tiếp hàng ngày.

### 🚀 Công nghệ chính
- **Ngôn ngữ:** Python
- **Framework Học sâu:** PyTorch
- **Trích xuất Đặc trưng:** MediaPipe
- **Xử lý Video & Ảnh:** OpenCV

---

## 2. Kiến trúc & Luồng hoạt động

Dự án áp dụng kiến trúc 2-giai-đoạn để đảm bảo hiệu quả và tốc độ xử lý:

### Giai đoạn 1: Trích xuất Đặc trưng (Feature Extraction)

Thay vì xử lý ảnh/video thô, hệ thống sử dụng thư viện **MediaPipe Holistic** của Google để trích xuất các điểm mốc (keypoints) quan trọng từ mỗi khung hình video.

- **Đầu vào:** Video một người thực hiện ký hiệu.
- **Quá trình:** MediaPipe phân tích và trả về tọa độ (x, y, z) của các điểm trên cơ thể, bàn tay và khuôn mặt.
- **Đầu ra:** Một chuỗi (sequence) các vector đặc trưng. Trong project này, mỗi video được chuẩn hóa thành một chuỗi có kích thước `(60, 201)`, trong đó:
    - `60` là số lượng khung hình (frames) sau khi chuẩn hóa.
    - `201` là số chiều đặc trưng cho mỗi khung hình (tọa độ của 67 keypoints x 3 chiều).

*Mô tả ảnh: Một người đang thực hiện ký hiệu, và trên cơ thể, bàn tay của họ có các điểm keypoints được MediaPipe nhận diện và nối lại với nhau.*

### Giai đoạn 2: Phân loại Ký hiệu (Action Classification)

Chuỗi vector đặc trưng từ Giai đoạn 1 được đưa vào một mô hình học sâu để phân loại và đoán ra đó là ký hiệu gì.

- **Lựa chọn mô hình:** **Bi-LSTM (Bidirectional Long Short-Term Memory)** được lựa chọn vì đây là kiến trúc phù hợp cho dữ liệu dạng chuỗi.
- **Lý do:** Bi-LSTM có khả năng ghi nhớ thông tin từ các khung hình trước đó (LSTM) và cả các khung hình sau đó (Bidirectional). Điều này giúp mô hình hiểu được toàn bộ ngữ cảnh của một động tác ký hiệu, thay vì chỉ nhìn vào từng khung hình riêng lẻ.

---

## 3. Quá trình Phát triển & Thách thức

### Giai đoạn 1: Xây dựng Pipeline với PyTorch

Em đã xây dựng thành công một pipeline hoàn chỉnh bằng PyTorch, bao gồm:

1.  **Tiền xử lý Dữ liệu (`preprocess.py`):** Script tự động chuyển đổi hàng nghìn video VSL thô thành các file dữ liệu `.npy` đã được chuẩn hóa về kích thước `(60, 201)`.
2.  **Tăng cường Dữ liệu (`augmentation.py`):** Xây dựng các hàm để xoay, dịch chuyển, phóng to/thu nhỏ chuỗi keypoints nhằm làm giàu dữ liệu huấn luyện và chống overfitting.
3.  **Custom Dataset (`dataset.py`):** Xây dựng class `SignLanguageDataset` kế thừa từ `torch.utils.data.Dataset` để tải dữ liệu `.npy` và áp dụng augmentation một cách hiệu quả.
    ```python
    class SignLanguageDataset(Dataset):
        def __init__(self, processed_data_dir, labels_csv_path, apply_augmentation=False):
            # ... đọc labels.csv và khởi tạo các thông số ...

        def __len__(self):
            return len(self.labels_df)

        def __getitem__(self, idx):
            # ... tải file .npy, lấy nhãn ...
            # ... nếu apply_augmentation=True, áp dụng các phép biến đổi ...
            return landmarks_tensor, label_id_tensor
    ```
4.  **Xây dựng Mô hình (`model.py`):** Triển khai kiến trúc Bi-LSTM sâu với các lớp Batch Normalization và Dropout như trong báo cáo nghiên cứu.
    ```python
    class BiLSTMClassifier(nn.Module):
        def __init__(self, num_classes, input_size=201):
            super(BiLSTMClassifier, self).__init__()
            # ... định nghĩa 3 lớp Bi-LSTM và các lớp Fully Connected ...

        def forward(self, x):
            # ... định nghĩa luồng dữ liệu đi qua các lớp ...
            return logits
    ```
5.  **Xây dựng Vòng lặp Huấn luyện (`train.py`):** Script hoàn chỉnh để load dữ liệu, khởi tạo model, huấn luyện và đánh giá trên tập validation, tích hợp cơ chế Early Stopping để tránh lãng phí thời gian huấn luyện.

### Giai đoạn 2: Thách thức trong Huấn luyện

Khi tiến hành huấn luyện mô hình PyTorch trên bộ dữ liệu đầy đủ (~3300 lớp), em đã gặp phải một thách thức lớn và rất phổ biến trong các bài toán thực tế:

- **Vấn đề:** Mô hình hội tụ rất chậm và độ chính xác trên tập validation rất thấp (Val Accuracy ~ 0%).
- **Nguyên nhân:** Dữ liệu quá thưa (sparse), với trung bình chỉ ~1.3 video cho mỗi ký hiệu. Với lượng dữ liệu ít ỏi cho mỗi lớp, mô hình không có đủ "ví dụ" để học và khái quát hóa, dẫn đến học vẹt (overfitting) hoặc không thể học được gì cả.

### Giai đoạn 3: Giải pháp "Chữa cháy" và Tích hợp

Để đảm bảo project có một sản phẩm hoạt động được và thể hiện khả năng giải quyết vấn đề, em đã đưa ra một giải pháp thực tiễn:

1.  **Quyết định:** Tận dụng model đã được huấn luyện sẵn từ các tác giả của bài báo cáo nghiên cứu. Model này có định dạng là `.keras`.
2.  **Thách thức Kỹ thuật:** Project đang được xây dựng bằng PyTorch, không thể sử dụng trực tiếp model `.keras` của TensorFlow.
3.  **Giải pháp:** Chuyển đổi model Keras sang định dạng **ONNX (Open Neural Network Exchange)**. ONNX là một định dạng model AI phổ thông, cho phép các framework khác nhau có thể sử dụng chung một model.
4.  **Kế hoạch tiếp theo:** Sau khi có file `model.onnx`, em sẽ sử dụng thư viện `onnxruntime` để load và chạy model này để thực hiện suy đoán (inference) trong môi trường Python.

---

## 4. Hướng phát triển

- **Trước mắt:** Hoàn thành việc chuyển đổi Keras -> ONNX và xây dựng script inference sử dụng file `.onnx`.
- **Tương lai:**
    1.  Xây dựng giao diện đồ họa (GUI) đơn giản bằng Gradio hoặc PyQt để demo khả năng nhận diện real-time qua webcam.
    2.  Quay lại với mô hình PyTorch, tìm cách thu thập thêm dữ liệu hoặc áp dụng các kỹ thuật huấn luyện nâng cao hơn (transfer learning, few-shot learning) để cải thiện độ chính xác.
    3.  Mở rộng bài toán từ nhận diện từ đơn lẻ sang nhận diện câu liên tục.

---

## 5. Kết luận

Qua quá trình thực hiện dự án, em đã:
- Xây dựng thành công một pipeline hoàn chỉnh từ dữ liệu thô đến mô hình học sâu bằng PyTorch.
- Tự mình chẩn đoán được một vấn đề huấn luyện rất thực tế (dữ liệu thưa).
- Thể hiện được khả năng giải quyết vấn đề bằng cách tìm tòi và áp dụng một giải pháp kỹ thuật phù hợp (sử dụng model pre-trained và chuyển đổi qua ONNX).

Dự án này là một kinh nghiệm quý báu, không chỉ về việc xây dựng mô hình AI mà còn về kỹ năng xử lý các vấn đề phát sinh trong một project kỹ thuật thực tế.
