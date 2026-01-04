# 📒 TÀI LIỆU CHI TIẾT DỰ ÁN: VIETNAMESE SIGN LANGUAGE (VSL) RECOGNITION

## 1. 🌟 Tổng Quan Dự Án
Dự án tập trung vào việc xây dựng một hệ thống trí tuệ nhân tạo (AI) có khả năng nhận diện và chuyển đổi Ngôn ngữ ký hiệu Việt Nam thành văn bản. Hệ thống hướng tới việc hỗ trợ cộng đồng người khiếm thính giao tiếp thuận tiện hơn trong đời sống và giáo dục.

## 2. 🛠️ Những Gì Đã Thực Hiện (Completed)

### 📥 Thu Thập & Xử Lý Dữ Liệu (Data Pipeline)
- **Crawl Dữ Liệu Tự Động:** Sử dụng Selenium để tải hàng nghìn video từ nguồn chính thống (Bộ Giáo dục & Đào tạo).
- **Trích Xuất Đặc Trưng (Feature Extraction):** Sử dụng **MediaPipe Holistic** để chuyển đổi video thành các điểm tọa độ (keypoints) của cơ thể và bàn tay (201 features/frame).
- **Chuẩn Hóa Thời Gian:** Đồng nhất độ dài mọi chuỗi hành động về **60 frames** bằng thuật toán nội suy Cubic.
- **Siêu Tăng Cường Dữ Liệu (Heavy Augmentation):** Áp dụng xoay, tịnh tiến, thay đổi tốc độ và đặc biệt là thuật toán **Inverse Kinematics (IK)** để thay đổi khoảng cách tay mà vẫn giữ cấu trúc cơ thể tự nhiên.

### 🧠 Kiến Trúc Mô Hình (Model Architecture)
- **Deep Bi-LSTM:** Sử dụng mạng nơ-ron hồi quy hai chiều 3 lớp để học các đặc trưng chuỗi thời gian phức tạp.
- **Tối Ưu Hóa:** Tích hợp Batch Normalization và Dropout để tăng tốc độ huấn luyện và chống Overfitting.
- **Quy Mô:** Hỗ trợ nhận diện lên tới **2764 nhãn** từ vựng khác nhau.

### 💻 Giao Diện Người Dùng (Frontend - Streamlit)
- **Dashboard Hiện Đại:** Phân chia Tab (Webcam/Video) và Sidebar cấu hình chuyên nghiệp.
- **Trải Nghiệm Camera (UX):**
    - Hiệu ứng gương (Mirror view) tự nhiên.
    - Đếm ngược trực quan (Ready 3-2-1) đè lên feed camera.
    - Chỉ báo ghi hình (REC) và thanh tiến trình (Progress bar) thời gian thực.
    - Tùy chọn ẩn/hiện khung xương (Landmarks) linh hoạt.
- **Hiển Thị Kết Quả:** Sử dụng biểu đồ Top-K xác suất để phân tích độ tin cậy của dự đoán.

## 3. 🚧 Những Gì Đang Thực Hiện (In Progress)
- **Tối Ưu Tốc Độ:** Giảm độ trễ xử lý MediaPipe trên các thiết bị cấu hình thấp (đã hạ độ phân giải và giới hạn buffer).
- **Hoàn Thiện UI:** Tinh chỉnh CSS để các thành phần hiển thị đồng nhất trên nhiều kích thước màn hình.

## 4. 💡 Định Hướng & Cải Thiện (Future Roadmap)

### 🚀 Cải Thiện Kỹ Thuật
- **Mô Hình Transformer:** Thử nghiệm kiến trúc Transformer thay cho LSTM để đạt độ chính xác cao hơn với các câu ký hiệu dài.
- **Nhận Diện Câu (Sentence Recognition):** Thay vì chỉ nhận diện từng từ đơn lẻ, phát triển mô hình có thể dịch cả một câu hoàn chỉnh.
- **Chuyển Đổi Sang TFLite:** Tối ưu hóa model để chạy trực tiếp trên trình duyệt web hoặc điện thoại di động mà không cần server mạnh.

### 🎨 Cải Thiện Trải Nghiệm (UX)
- **Text-to-Speech (TTS):** Thêm tính năng phát âm thanh cho từ vựng vừa dự đoán được.
- **Lịch Sử Dự Đoán:** Lưu lại các từ đã nhận diện để ghép thành một đoạn hội thoại hoàn chỉnh.
- **Chế Độ Học Tập:** Thêm module cho phép người dùng nhìn vào mẫu và tập thực hiện theo, hệ thống sẽ chấm điểm độ chính xác.

### 🌏 Mở Rộng Dữ Liệu
- **Dữ Liệu Cộng Đồng:** Cho phép người dùng đóng góp video ký hiệu cá nhân để đa dạng hóa tập mẫu (vùng miền, độ tuổi).

---
*Tài liệu được cập nhật ngày: 04/01/2026*
