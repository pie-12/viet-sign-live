# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm # Thư viện để tạo thanh tiến trình (progress bar)
import numpy as np # Import numpy for best_val_loss initialization

from data.dataset import SignLanguageDataset
# from data.utils import pad_collate_fn # Không còn cần thiết
from models.model import BiLSTMClassifier

# --- 1. CẤU HÌNH & SIÊU THAM SỐ (HYPERPARAMETERS) ---

# Đường dẫn dữ liệu
# Dữ liệu nằm trong thư mục 'data' của dự án
# For Kaggle, point to the absolute path of the input dataset
KAGGLE_DATA_ROOT = '/kaggle/input/viet-sign-language-data'
if os.path.exists(KAGGLE_DATA_ROOT):
    PROCESSED_DATA_DIR = os.path.join(KAGGLE_DATA_ROOT, 'processed_data_60_201')
    LABELS_CSV_PATH = os.path.join(KAGGLE_DATA_ROOT, 'labels.csv')
else: # Local fallback
    PROCESSED_DATA_DIR = os.path.join('data', 'processed_data_60_201')
    LABELS_CSV_PATH = os.path.join('data', 'labels.csv')

# Siêu tham số cho quá trình huấn luyện
NUM_EPOCHS = 100      # Số epoch tối đa
BATCH_SIZE = 32       # Kích thước batch, như trong báo cáo
LEARNING_RATE = 0.001 # Tốc độ học, như trong báo cáo
VALIDATION_SPLIT = 0.2 # Tỷ lệ dữ liệu dùng cho validation (20%)

# Early Stopping (như trong báo cáo)
EARLY_STOPPING_PATIENCE = 10 # Dừng nếu val loss không cải thiện sau 10 epoch
# For Kaggle, save to the writable /kaggle/working/ directory
if os.path.exists('/kaggle/working/'):
    MODEL_SAVE_PATH = "/kaggle/working/viet_sign_live_bilstm_best_model.pth"
else: # Local fallback
    MODEL_SAVE_PATH = "viet_sign_live_bilstm_best_model.pth" # Tên file để lưu mô hình tốt nhất

def train():
    print("--- Bắt đầu quá trình huấn luyện ---")

    # --- 2. THIẾT LẬP THIẾT BỊ (CPU/GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 3. CHUẨN BỊ DỮ LIỆU ---
    print("Đang tải và chuẩn bị dữ liệu...")
    
    # Dataset đầy đủ (chưa chia train/val, không augmentation) để lấy thông tin tổng quan
    full_dataset_info = SignLanguageDataset(
        processed_data_dir=PROCESSED_DATA_DIR, 
        labels_csv_path=LABELS_CSV_PATH,
        apply_augmentation=False # Không augmentation để tránh tạo ra dữ liệu trùng lặp khi chia
    )
    
    num_classes = len(full_dataset_info.label_to_id)
    print(f"Tổng số lớp (ký hiệu): {num_classes}")

    # Chia dataset thành tập training và validation
    val_size = int(len(full_dataset_info) * VALIDATION_SPLIT)
    train_size = len(full_dataset_info) - val_size
    
    # random_split trả về các chỉ mục, không phải các instance dataset riêng biệt
    # Do đó, cần tạo lại dataset cho train và val với apply_augmentation phù hợp
    indices = list(range(len(full_dataset_info)))
    np.random.shuffle(indices) # Đảm bảo chia ngẫu nhiên
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Tạo dataset cho training (có augmentation) và validation (không augmentation)
    train_full_dataset = SignLanguageDataset(
        processed_data_dir=PROCESSED_DATA_DIR,
        labels_csv_path=LABELS_CSV_PATH,
        apply_augmentation=True # Bật augmentation cho tập huấn luyện
    )
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)

    val_full_dataset = SignLanguageDataset(
        processed_data_dir=PROCESSED_DATA_DIR,
        labels_csv_path=LABELS_CSV_PATH,
        apply_augmentation=False # Không augmentation cho tập validation
    )
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)
    
    print(f"Kích thước tập training: {len(train_dataset)}")
    print(f"Kích thước tập validation: {len(val_dataset)}")

    # Tạo DataLoader cho training và validation (không cần collate_fn tùy chỉnh nữa)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 4. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER ---
    print("Đang khởi tạo mô hình...")
    # Mô hình BiLSTMClassifier mới chỉ cần num_classes
    model = BiLSTMClassifier(num_classes=num_classes).to(device)

    # Loss function (Hàm mất mát) - CrossEntropyLoss đã bao gồm softmax
    criterion = nn.CrossEntropyLoss()

    # Optimizer (Bộ tối ưu hóa) - Adam như trong báo cáo
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. VÒNG LẶP HUẤN LUYỆN ---
    best_val_loss = float('inf') # Theo dõi validation loss tốt nhất
    patience_counter = 0         # Bộ đếm cho Early Stopping

    for epoch in range(NUM_EPOCHS):
        # ** Giai đoạn Training **
        model.train() # Đặt mô hình ở chế độ training
        running_train_loss = 0.0
        
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

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

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_predictions) * 100

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Đã lưu mô hình mới với Val Loss tốt hơn: {best_val_loss:.4f} tại '{MODEL_SAVE_PATH}'")
        else:
            patience_counter += 1
            print(f"Val Loss không cải thiện. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Dừng sớm: Val Loss không cải thiện sau {EARLY_STOPPING_PATIENCE} epoch.")
                break

    print("--- Quá trình huấn luyện hoàn tất ---")
    print(f"Mô hình tốt nhất đã được lưu tại '{MODEL_SAVE_PATH}' với Val Loss tốt nhất là {best_val_loss:.4f}")


if __name__ == "__main__":
    # Cài đặt thư viện tqdm nếu chưa có (đã có trong requirements.txt)
    train()
