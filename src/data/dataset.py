# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        """
        Khởi tạo Dataset.
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa các file .npy (processed_data).
            labels_file (str): Đường dẫn đến file labels.csv.
            transform (callable, optional): Một hàm biến đổi tùy chọn để áp dụng cho dữ liệu.
        """
        self.data_dir = data_dir
        full_labels_df = pd.read_csv(labels_file)
        self.transform = transform

        # --- LỌC DATASET ĐỂ GIỮ LẠI N LỚP PHỔ BIẾN NHẤT ---
        NUM_CLASSES_TO_KEEP = 10 # Bắt đầu với 10 lớp
        print(f"--- LƯU Ý: Chỉ giữ lại {NUM_CLASSES_TO_KEEP} lớp (ký hiệu) phổ biến nhất để huấn luyện ---")

        # Tìm N lớp phổ biến nhất
        top_labels = full_labels_df['label'].value_counts().nlargest(NUM_CLASSES_TO_KEEP).index.tolist()

        # Lọc dataframe để chỉ chứa các mẫu thuộc các lớp này
        self.labels_df = full_labels_df[full_labels_df['label'].isin(top_labels)].reset_index(drop=True)
        
        print(f"Các lớp được giữ lại: {top_labels}")
        print(f"Số lượng mẫu sau khi lọc: {len(self.labels_df)}")


        # Ánh xạ các nhãn (tên ký hiệu) thành các số nguyên (ID)
        # Ví dụ: "XIN CHAO" -> 0, "CAM ON" -> 1, ...
        self.label_to_id = {label: i for i, label in enumerate(self.labels_df['label'].unique())}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        print(f"Đã tải {len(self.labels_df)} mẫu dữ liệu từ {labels_file}")
        print(f"Tổng số lớp (ký hiệu) duy nhất: {len(self.label_to_id)}")
        # print(f"Ánh xạ nhãn sang ID: {self.label_to_id}") # Có thể bỏ comment để xem ánh xạ

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
        mp4_file_name = row['filename']  # Lấy tên file .mp4 từ cột 'filename'
        label_str = row['label']         # Nhãn dạng chuỗi (ví dụ: "XIN CHAO")

        # Thay đổi đuôi file từ .mp4 thành .npy
        npy_file_name = mp4_file_name.replace('.mp4', '.npy')

        # Xây dựng đường dẫn đầy đủ đến file .npy
        npy_file_path = os.path.join(self.data_dir, npy_file_name)

        # Tải dữ liệu landmarks từ file .npy
        # Dữ liệu này có dạng (số_frame, số_landmark, số_tọa_độ_xyz)
        landmarks = np.load(npy_file_path)

        # Chuyển đổi nhãn chuỗi thành ID số nguyên
        label_id = self.label_to_id[label_str]

        # Chuyển đổi dữ liệu numpy thành PyTorch Tensor
        # torch.float32 là kiểu dữ liệu chuẩn cho đầu vào mô hình
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        # Nhãn thường là số nguyên, nên dùng torch.long
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)

        # Áp dụng các biến đổi (transform) nếu có
        if self.transform:
            landmarks_tensor = self.transform(landmarks_tensor)

        return landmarks_tensor, label_id_tensor