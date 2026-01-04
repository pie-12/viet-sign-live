# src/models/model.py

import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, num_classes, input_size=201):
        """
        Khởi tạo mô hình Bi-LSTM sâu để phân loại ngôn ngữ ký hiệu dựa trên kiến trúc trong báo cáo.

        Args:
            num_classes (int): Số lượng lớp đầu ra (số lượng ký hiệu duy nhất).
            input_size (int): Kích thước của mỗi frame đầu vào (số lượng đặc trưng landmarks). Mặc định là 201.
        """
        super(BiLSTMClassifier, self).__init__()

        # Các hằng số cho kiến trúc mô hình
        lstm_hidden_size = 256
        lstm_dropout = 0.3
        dense1_units = 512
        dense2_units = 256
        dense_dropout = 0.5

        # --- Lớp Bi-LSTM đầu tiên ---
        self.lstm1 = nn.LSTM(input_size, lstm_hidden_size, num_layers=1,
                             batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size * 2) # *2 vì Bi-directional

        # --- Lớp Bi-LSTM thứ hai ---
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, num_layers=1,
                             batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.bn2 = nn.BatchNorm1d(lstm_hidden_size * 2)

        # --- Lớp Bi-LSTM thứ ba (không trả về sequence) ---
        self.lstm3 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, num_layers=1,
                             batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.bn3 = nn.BatchNorm1d(lstm_hidden_size * 2) # Output của lstm3 là (batch_size, hidden_size * 2)

        # --- Các lớp Dense ---
        self.fc1 = nn.Linear(lstm_hidden_size * 2, dense1_units)
        self.bn4 = nn.BatchNorm1d(dense1_units)
        self.dropout1 = nn.Dropout(dense_dropout)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(dense1_units, dense2_units)
        self.bn5 = nn.BatchNorm1d(dense2_units)
        self.dropout2 = nn.Dropout(dense_dropout)
        self.relu2 = nn.ReLU()

        # --- Lớp đầu ra ---
        self.output_layer = nn.Linear(dense2_units, num_classes)

    def forward(self, x):
        """
        Phương thức forward định nghĩa cách dữ liệu đi qua mô hình.

        Args:
            x (torch.Tensor): Tensor đầu vào có dạng (batch_size, sequence_length=60, input_size=201).

        Returns:
            torch.Tensor: Logits đầu ra có dạng (batch_size, num_classes).
        """
        batch_size, seq_len, _ = x.size()

        # --- Qua Bi-LSTM 1 ---
        # input: (batch, seq_len, input_size) -> (batch, seq_len, hidden_size * 2)
        out1, _ = self.lstm1(x)
        # BatchNorm1d mong đợi đầu vào (batch_size, features, sequence_length)
        # Chúng ta có (batch_size, sequence_length, features) -> transpose để phù hợp
        out1 = self.bn1(out1.transpose(1, 2)).transpose(1, 2)
        
        # --- Qua Bi-LSTM 2 ---
        out2, _ = self.lstm2(out1)
        out2 = self.bn2(out2.transpose(1, 2)).transpose(1, 2)

        # --- Qua Bi-LSTM 3 ---
        # Lớp này không trả về sequence, mà chỉ trả về output cuối cùng
        # out3: (batch_size, hidden_size * 2)
        out3, (hn, cn) = self.lstm3(out2)
        # Lấy hidden state cuối cùng từ cả hai chiều
        # hn có dạng (num_layers * num_directions, batch_size, hidden_size)
        # Vì num_layers=1 cho mỗi lstm, chúng ta lấy hn[0] (forward) và hn[1] (backward)
        # Nối chúng lại
        final_lstm_output = torch.cat((hn[0, :, :], hn[1, :, :]), dim=1)
        
        # BatchNorm3d cho tensor 2D (batch_size, features)
        final_lstm_output = self.bn3(final_lstm_output)
        
        # --- Qua các lớp Dense ---
        out = self.fc1(final_lstm_output)
        out = self.bn4(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn5(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # --- Lớp đầu ra ---
        logits = self.output_layer(out)

        return logits
