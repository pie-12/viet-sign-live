# src/models/model.py

import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        """
        Khởi tạo mô hình Bi-LSTM để phân loại ngôn ngữ ký hiệu.

        Args:
            input_size (int): Kích thước của mỗi frame đầu vào (số lượng đặc trưng landmarks).
                              Trong trường hợp của chúng ta là 258.
            hidden_size (int): Số lượng đặc trưng trong trạng thái ẩn (hidden state) của LSTM.
                               Đây là một siêu tham số (hyperparameter) có thể điều chỉnh.
            num_layers (int): Số lượng lớp LSTM xếp chồng lên nhau.
            num_classes (int): Số lượng lớp đầu ra (số lượng ký hiệu duy nhất).
                               Trong trường hợp của chúng ta là 3315.
            dropout_rate (float): Tỷ lệ dropout để chống overfitting.
        """
        super(BiLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Lớp Bi-LSTM
        # batch_first=True: Đầu vào và đầu ra sẽ có dạng (batch_size, sequence_length, features)
        # bidirectional=True: Sử dụng Bi-LSTM (chạy cả hai chiều)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # Lớp Linear (fully connected layer) để phân loại đầu ra
        # Vì là Bi-LSTM, hidden state cuối cùng sẽ có kích thước gấp đôi (hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # Lớp Dropout (để chống overfitting)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Phương thức forward định nghĩa cách dữ liệu đi qua mô hình.

        Args:
            x (torch.Tensor): Tensor đầu vào có dạng (batch_size, sequence_length, input_size).
                              Đây chính là landmarks_batch đã được đệm từ DataLoader.

        Returns:
            torch.Tensor: Logits đầu ra có dạng (batch_size, num_classes).
        """
        # Khởi tạo trạng thái ẩn và trạng thái tế bào ban đầu (hidden state và cell state)
        # cho LSTM. Thường khởi tạo bằng 0.
        # hidden_state có dạng (num_layers * num_directions, batch_size, hidden_size)
        # num_directions = 2 vì chúng ta dùng Bi-LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Truyền dữ liệu qua lớp LSTM
        # out: output features from the last layer of the LSTM (batch_size, sequence_length, num_directions * hidden_size)
        # (hn, cn): final hidden state and cell state for each layer
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Lấy hidden state của lớp cuối cùng (last layer) và chiều cuối cùng (last direction)
        # hn có dạng (num_layers * num_directions, batch_size, hidden_size)
        # Chúng ta cần hidden state của lớp cuối cùng (num_layers - 1) và cả hai chiều (forward và backward)
        # Để lấy hidden state của lớp cuối cùng từ cả hai chiều:
        # hn[-2, :, :] là hidden state của lớp cuối cùng, chiều forward
        # hn[-1, :, :] là hidden state của lớp cuối cùng, chiều backward
        # Chúng ta nối chúng lại (concatenate)
        final_hidden_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        # Áp dụng Dropout
        final_hidden_state = self.dropout(final_hidden_state)

        # Truyền qua lớp Linear để có logits đầu ra
        logits = self.fc(final_hidden_state)

        return logits
