import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# --- Cấu hình ---
# Đường dẫn đến thư mục chứa dataset đã tải về
DATA_DIR = r"D:\Dataset\VietSignLive"
VIDEO_DIR = os.path.join(DATA_DIR, "videos_mp4")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")

# Thư mục để lưu dữ liệu đã qua xử lý (các file .npy)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data")

def extract_landmarks(video_path, holistic):
    """
    Trích xuất các điểm mốc từ một video và trả về một mảng numpy.
    Mỗi frame sẽ là một hàng trong mảng.
    """
    landmarks_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển màu từ BGR sang RGB để MediaPipe xử lý
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Thực hiện nhận diện
        results = holistic.process(image)

        image.flags.writeable = True

        # Trích xuất các điểm mốc và làm phẳng (flatten)
        # Pose: 33 landmarks * 4 giá trị (x, y, z, visibility) = 132
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        # Left Hand: 21 landmarks * 3 giá trị (x, y, z) = 63
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        # Right Hand: 21 landmarks * 3 giá trị (x, y, z) = 63
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        # Tổng cộng: 132 + 63 + 63 = 258 giá trị cho mỗi frame
        frame_landmarks = np.concatenate([pose, lh, rh])
        landmarks_list.append(frame_landmarks)

    cap.release()
    return np.array(landmarks_list)

def main():
    """
    Hàm chính để xử lý tất cả các video trong dataset.
    """
    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Khởi tạo mô hình MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Đọc file labels.csv để lấy danh sách video cần xử lý
    try:
        with open(LABELS_FILE, 'r', newline='', encoding='utf-8') as csvfile:
            # Sử dụng DictReader để dễ dàng truy cập cột bằng tên
            reader = csv.DictReader(csvfile)
            video_files = list(reader)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {LABELS_FILE}. Hãy chắc chắn file downloader.py đã chạy thành công.")
        return

    print(f"Bắt đầu xử lý và trích xuất đặc trưng cho {len(video_files)} video...")

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for item in tqdm(video_files, desc="Đang xử lý video"):
        video_filename = item['filename']
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # Tên file output (ví dụ: D0001.mp4 -> D0001.npy)
        npy_filename = os.path.splitext(video_filename)[0] + ".npy"
        npy_path = os.path.join(PROCESSED_DIR, npy_filename)

        # Bỏ qua nếu file đã được xử lý
        if os.path.exists(npy_path):
            continue

        # Trích xuất đặc trưng
        landmarks = extract_landmarks(video_path, holistic)

        # Lưu lại dưới dạng file numpy
        if landmarks is not None and landmarks.shape[0] > 0: # Chỉ lưu nếu có frame được xử lý
            np.save(npy_path, landmarks)

    holistic.close()
    print("\nHoàn tất! Dữ liệu đã được xử lý và lưu tại thư mục 'processed_data'.")
    print("Bước tiếp theo là chuẩn bị dữ liệu này để huấn luyện mô hình PyTorch.")


if __name__ == "__main__":
    # Trước khi chạy, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết:
    # pip install opencv-python mediapipe numpy
    main()
