# test_dataloader.py

from dataset import SignLanguageDataset
from torch.utils.data import DataLoader
import os
import torch

from torch.nn.utils.rnn import pad_sequence

# --- Cấu hình đường dẫn dữ liệu của bạn ---
DATA_DIR = r"D:\Dataset\VietSignLive\processed_data"
LABELS_FILE = r"D:\Dataset\VietSignLive\labels.csv"
# ------------------------------------------

def pad_collate_fn(batch):
    """
    Hàm collate tùy chỉnh để đệm (pad) các chuỗi trong một batch.
    Args:
        batch (list): Một list các tuple, mỗi tuple là (landmarks_tensor, label_id_tensor).
    Returns:
        tuple: (padded_landmarks, labels)
    """
    # Tách riêng landmarks và labels từ batch
    landmarks_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Đệm các chuỗi landmarks
    # batch_first=True nghĩa là tensor đầu ra sẽ có shape (batch_size, seq_len, features)
    padded_landmarks = pad_sequence(landmarks_list, batch_first=True, padding_value=0.0)

    # Gom các labels lại thành một tensor
    labels = torch.stack(labels_list)

    return padded_landmarks, labels

def test_dataloader():
    print("--- Bắt đầu kiểm tra PyTorch DataLoader ---")

    # 1. Khởi tạo Dataset
    try:
        dataset = SignLanguageDataset(data_dir=DATA_DIR, labels_file=LABELS_FILE)
        print(f"Dataset đã được khởi tạo thành công với {len(dataset)} mẫu.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo Dataset: {e}")
        return

    # 2. Khởi tạo DataLoader
    batch_size = 32
    shuffle = True
    num_workers = 0

    try:
        # Sử dụng hàm collate tùy chỉnh của chúng ta
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn)
        print(f"DataLoader đã được khởi tạo thành công với batch_size = {batch_size} và collate_fn tùy chỉnh.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo DataLoader: {e}")
        return

    # 3. Lặp qua một vài batch và in kích thước của chúng
    print("\n--- Lặp qua các batch từ DataLoader ---")
    for i, (landmarks_batch, labels_batch) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        # Giờ đây, kích thước của landmarks_batch sẽ đồng nhất trong mỗi batch
        print(f"  Kích thước của landmarks_batch (đã đệm): {landmarks_batch.shape}")
        print(f"  Kích thước của labels_batch: {labels_batch.shape}")

        # In ra một vài thông tin chi tiết hơn về batch đầu tiên
        if i == 0:
            print(f"  Kiểu dữ liệu của landmarks_batch: {landmarks_batch.dtype}")
            print(f"  Kiểu dữ liệu của labels_batch: {labels_batch.dtype}")
            print(f"  Ví dụ nhãn trong batch đầu tiên: {labels_batch[:5].tolist()}")
            print(f"  Nhãn gốc (chuỗi) tương ứng: {[dataset.id_to_label[id.item()] for id in labels_batch[:5]]}")

        # Chỉ lấy 3 batch đầu tiên để kiểm tra
        if i >= 2:
            break

    print("\n--- Kiểm tra PyTorch DataLoader hoàn tất ---")

if __name__ == "__main__":
    test_dataloader()
