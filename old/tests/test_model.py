# tests/test_model.py

import torch
from src.models.model import BiLSTMClassifier
from src.data.dataset import SignLanguageDataset # To get num_classes
import os

# --- Cấu hình đường dẫn dữ liệu của bạn ---
LABELS_FILE = r"D:\Dataset\VietSignLive\labels.csv"
# ------------------------------------------

def test_model():
    print("--- Bắt đầu kiểm tra BiLSTMClassifier ---")

    # 1. Lấy số lượng lớp (num_classes) từ Dataset
    try:
        # Khởi tạo Dataset chỉ để lấy số lượng lớp
        # data_dir không cần thiết ở đây vì chúng ta chỉ cần labels_file
        temp_dataset = SignLanguageDataset(data_dir="", labels_file=LABELS_FILE)
        num_classes = len(temp_dataset.label_to_id)
        print(f"Số lượng lớp (ký hiệu) từ Dataset: {num_classes}")
    except Exception as e:
        print(f"Lỗi khi lấy số lượng lớp từ Dataset: {e}")
        return

    # 2. Định nghĩa các tham số cho mô hình
    input_size = 258  # Kích thước của mỗi frame landmarks (từ landmarks_tensor.shape[1])
    hidden_size = 128 # Kích thước hidden state của LSTM
    num_layers = 2    # Số lớp LSTM
    dropout_rate = 0.5

    print(f"Khởi tạo mô hình với: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, num_classes={num_classes}")

    # 3. Khởi tạo mô hình
    try:
        model = BiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_rate)
        print("Mô hình BiLSTMClassifier đã được khởi tạo thành công.")
        # print(model) # Có thể bỏ comment để xem cấu trúc chi tiết của mô hình
    except Exception as e:
        print(f"Lỗi khi khởi tạo mô hình: {e}")
        return

    # 4. Tạo một tensor đầu vào giả (dummy input)
    # Dạng: (batch_size, sequence_length, input_size)
    # Giả sử batch_size = 4, sequence_length = 150 (một độ dài chuỗi điển hình)
    batch_size = 4
    sequence_length = 150
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    print(f"Tạo tensor đầu vào giả với kích thước: {dummy_input.shape}")

    # 5. Truyền đầu vào giả qua mô hình
    print("Truyền đầu vào giả qua mô hình...")
    try:
        output = model(dummy_input)
        print("Mô hình đã xử lý đầu vào thành công.")
        print(f"Kích thước đầu ra của mô hình: {output.shape}")

        # Kiểm tra kích thước đầu ra
        expected_output_shape = (batch_size, num_classes)
        if output.shape == expected_output_shape:
            print(f"Kích thước đầu ra khớp với mong đợi: {expected_output_shape}. OK.")
        else:
            print(f"Lỗi: Kích thước đầu ra không khớp. Mong đợi {expected_output_shape}, nhận được {output.shape}.")

    except Exception as e:
        print(f"Có lỗi xảy ra khi truyền đầu vào qua mô hình: {e}")
        return

    print("\n--- Kiểm tra BiLSTMClassifier hoàn tất ---")

if __name__ == "__main__":
    test_model()
