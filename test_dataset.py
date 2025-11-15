# test_dataset.py

from dataset import SignLanguageDataset
import os
import torch

# --- Cấu hình đường dẫn dữ liệu của bạn ---
# Đảm bảo các đường dẫn này chính xác với máy của bạn
DATA_DIR = r"D:\Dataset\VietSignLive\processed_data"
LABELS_FILE = r"D:\Dataset\VietSignLive\labels.csv"
# ------------------------------------------

def test_dataset():
    print("--- Bắt đầu kiểm tra SignLanguageDataset ---")

    # 1. Khởi tạo Dataset
    try:
        dataset = SignLanguageDataset(data_dir=DATA_DIR, labels_file=LABELS_FILE)
        print(f"Dataset đã được khởi tạo thành công với {len(dataset)} mẫu.")
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file hoặc thư mục. Vui lòng kiểm tra lại đường dẫn:")
        print(f"  DATA_DIR: {DATA_DIR}")
        print(f"  LABELS_FILE: {LABELS_FILE}")
        print(f"Chi tiết lỗi: {e}")
        return
    except Exception as e:
        print(f"Có lỗi xảy ra khi khởi tạo Dataset: {e}")
        return

    # 2. Kiểm tra độ dài của Dataset
    if len(dataset) == 0:
        print("Cảnh báo: Dataset rỗng. Vui lòng kiểm tra file labels.csv và thư mục processed_data.")
        return

    # 3. Lấy một mẫu dữ liệu đầu tiên và in thông tin
    print("\n--- Lấy mẫu dữ liệu đầu tiên (index 0) ---")
    try:
        landmarks_tensor, label_id_tensor = dataset[0]
        print(f"Kích thước của landmarks_tensor: {landmarks_tensor.shape}")
        print(f"Kiểu dữ liệu của landmarks_tensor: {landmarks_tensor.dtype}")
        print(f"Giá trị của label_id_tensor: {label_id_tensor.item()}")
        print(f"Nhãn gốc (chuỗi) tương ứng: {dataset.id_to_label[label_id_tensor.item()]}")

        # Kiểm tra xem tensor có phải là PyTorch Tensor không
        if isinstance(landmarks_tensor, torch.Tensor) and isinstance(label_id_tensor, torch.Tensor):
            print("Dữ liệu được trả về là PyTorch Tensor. OK.")
        else:
            print("Lỗi: Dữ liệu trả về không phải là PyTorch Tensor.")

    except IndexError:
        print("Lỗi: Không thể lấy mẫu dữ liệu từ dataset (có thể dataset rỗng).")
    except Exception as e:
        print(f"Có lỗi xảy ra khi lấy mẫu dữ liệu: {e}")

    print("\n--- Kiểm tra SignLanguageDataset hoàn tất ---")

if __name__ == "__main__":
    test_dataset()
