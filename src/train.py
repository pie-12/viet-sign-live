# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm # Thư viện để tạo thanh tiến trình (progress bar)

from data.dataset import SignLanguageDataset
from data.utils import pad_collate_fn
from models.model import BiLSTMClassifier

# --- 1. CẤU HÌNH & SIÊU THAM SỐ (HYPERPARAMETERS) ---

# Đường dẫn dữ liệu
DATA_DIR = "data/VietSignLive/processed_data"
LABELS_FILE = "data/VietSignLive/labels.csv"

# Siêu tham số cho mô hình
INPUT_SIZE = 258      # Số đặc trưng landmarks cho mỗi frame
HIDDEN_SIZE = 256     # Kích thước của hidden state trong LSTM
NUM_LAYERS = 2        # Số lớp LSTM xếp chồng
DROPOUT_RATE = 0.5    # Tỷ lệ dropout

# Siêu tham số cho quá trình huấn luyện
NUM_EPOCHS = 50       # Số lần lặp qua toàn bộ dataset
BATCH_SIZE = 64       # Số lượng mẫu trong một batch
LEARNING_RATE = 0.001 # Tốc độ học của optimizer
VALIDATION_SPLIT = 0.2 # Tỷ lệ dữ liệu dùng cho validation (20%)
MODEL_SAVE_PATH = "viet_sign_live_bilstm.pth" # Tên file để lưu mô hình

def train():
    print("--- Bắt đầu quá trình huấn luyện ---")

    # --- 2. THIẾT LẬP THIẾT BỊ (CPU/GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 3. CHUẨN BỊ DỮ LIỆU ---
    print("Đang tải và chuẩn bị dữ liệu...")
    full_dataset = SignLanguageDataset(data_dir=DATA_DIR, labels_file=LABELS_FILE)
    
    # Lấy số lượng lớp từ dataset
    num_classes = len(full_dataset.label_to_id)
    print(f"Tổng số lớp (ký hiệu): {num_classes}")

    # Chia dataset thành tập training và validation
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Kích thước tập training: {len(train_dataset)}")
    print(f"Kích thước tập validation: {len(val_dataset)}")

    # Tạo DataLoader cho training và validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn, num_workers=2)

    # --- 4. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER ---
    print("Đang khởi tạo mô hình...")
    model = BiLSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # Loss function (Hàm mất mát)
    criterion = nn.CrossEntropyLoss()

    # Optimizer (Bộ tối ưu hóa)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. VÒNG LẶP HUẤN LUYỆN ---
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        # ** Giai đoạn Training **
        model.train() # Đặt mô hình ở chế độ training
        running_train_loss = 0.0
        
        # Sử dụng tqdm để tạo thanh tiến trình
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for inputs, labels in train_progress_bar:
            # --- DATA AUGMENTATION: Thêm nhiễu ngẫu nhiên ---
            # Chỉ áp dụng trong quá trình training
            noise = torch.randn(inputs.size()) * 0.01  # Tạo nhiễu nhỏ
            inputs = inputs + noise
            # ---------------------------------------------

            # Chuyển dữ liệu lên thiết bị (CPU/GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 1. Zero the parameter gradients (Reset gradient)
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 3. Backward pass
            loss.backward()

            # 4. Optimize (Cập nhật trọng số)
            optimizer.step()

            running_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)

        # ** Giai đoạn Validation **
        model.eval() # Đặt mô hình ở chế độ evaluation
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        with torch.no_grad(): # Không cần tính gradient trong giai đoạn validation
            for inputs, labels in val_progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Tính accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                val_progress_bar.set_postfix(loss=loss.item())

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_predictions) * 100

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # Lưu lại mô hình nếu có kết quả validation tốt hơn
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Đã lưu mô hình mới với Val Accuracy tốt hơn: {best_val_accuracy:.2f}% tại '{MODEL_SAVE_PATH}'")

    print("--- Quá trình huấn luyện hoàn tất ---")
    print(f"Mô hình đã được lưu tại '{MODEL_SAVE_PATH}' với Val Accuracy tốt nhất là {best_val_accuracy:.2f}%")


if __name__ == "__main__":
    # Cài đặt thư viện tqdm nếu chưa có
    try:
        # import tqdm # Dòng này gây lỗi, đã được xóa
        pass
    except ImportError:
        print("Đang cài đặt thư viện 'tqdm' để hiển thị thanh tiến trình...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

    train()
