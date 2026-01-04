# src/data/utils.py

import torch
from torch.nn.utils.rnn import pad_sequence

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
