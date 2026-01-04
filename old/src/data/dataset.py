# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import csv # Add csv import
from .augmentation import augment_sequence # Import augmentation functions

class SignLanguageDataset(Dataset):
    def __init__(self, processed_data_dir, labels_csv_path, apply_augmentation=False):
        """
        Khởi tạo Dataset.
        Args:
            processed_data_dir (str): Đường dẫn đến thư mục chứa các file .npy đã tiền xử lý (processed_data_60_201).
            labels_csv_path (str): Đường dẫn đến file labels.csv gốc.
            apply_augmentation (bool): Có áp dụng tăng cường dữ liệu hay không.
        """
        self.processed_data_dir = processed_data_dir
        self.apply_augmentation = apply_augmentation

        # Đọc file labels.csv gốc để có ánh xạ filename -> label
        original_labels_df = pd.read_csv(labels_csv_path)
        
        # Tạo danh sách các mẫu dữ liệu từ thư mục processed_data_60_201
        data_samples = []
        for npy_file in os.listdir(processed_data_dir):
            if npy_file.endswith('.npy'):
                original_mp4_filename = npy_file.replace('.npy', '.mp4')
                
                # Tìm label tương ứng từ original_labels_df
                # Đảm bảo cột 'filename' trong labels.csv chứa tên file .mp4
                label_row = original_labels_df[original_labels_df['filename'] == original_mp4_filename]
                
                if not label_row.empty:
                    label_str = label_row['label'].iloc[0]
                    data_samples.append({
                        'npy_filename': npy_file,
                        'label': label_str
                    })
                # else:
                    # print(f"Cảnh báo: Không tìm thấy nhãn cho file {original_mp4_filename}")

        self.labels_df = pd.DataFrame(data_samples)

        # Ánh xạ các nhãn (tên ký hiệu) thành các số nguyên (ID)
        self.label_to_id = {label: i for i, label in enumerate(self.labels_df['label'].unique())}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        print(f"Đã tải {len(self.labels_df)} mẫu dữ liệu đã tiền xử lý từ {processed_data_dir}")
        print(f"Tổng số lớp (ký hiệu) duy nhất: {len(self.label_to_id)}")

    def __len__(self):
        """
        Trả về tổng số lượng mẫu trong dataset.
        """
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu tại chỉ số (index) 'idx'.
        Args:
            idx (int): Chỉ số của mẫu dữ liệu cần lấy.
        Returns:
            tuple: (landmarks_tensor, label_id_tensor)
        """
        # Lấy thông tin về mẫu dữ liệu từ DataFrame
        row = self.labels_df.iloc[idx]
        npy_file_name = row['npy_filename']
        label_str = row['label']

        # Xây dựng đường dẫn đầy đủ đến file .npy
        npy_file_path = os.path.join(self.processed_data_dir, npy_file_name)

        # Tải dữ liệu landmarks từ file .npy
        # Dữ liệu này có dạng (60, 201)
        landmarks = np.load(npy_file_path)

        # Áp dụng tăng cường dữ liệu nếu được yêu cầu
        if self.apply_augmentation:
            landmarks = augment_sequence(landmarks) # num_augmentations được xử lý bên trong augment_sequence

        # Chuyển đổi nhãn chuỗi thành ID số nguyên
        label_id = self.label_to_id[label_str]

        # Chuyển đổi dữ liệu numpy thành PyTorch Tensor
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)

        return landmarks_tensor, label_id_tensor