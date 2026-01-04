import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# --- Cấu hình ---
# Đường dẫn đến thư mục chứa dataset đã tải về
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
VIDEO_DIR = os.path.join(DATA_DIR, "videos_mp4")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")

# Thư mục để lưu dữ liệu đã qua xử lý (các file .npy)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data_60_201")
TARGET_FRAMES = 60 # Số frame mục tiêu cho mỗi video

# Số lượng keypoints và chiều
# Pose (upper body): 25 landmarks * 3 coords (x,y,z) = 75
# Hands: 21 landmarks * 3 coords (x,y,z) * 2 (hai tay) = 126
# Total: 75 + 126 = 201
NUM_POSE_LANDMARKS = 25
NUM_HAND_LANDMARKS = 21
POSE_DIMS = NUM_POSE_LANDMARKS * 3
LH_DIMS = NUM_HAND_LANDMARKS * 3
RH_DIMS = NUM_HAND_LANDMARKS * 3
TOTAL_DIMS = POSE_DIMS + LH_DIMS + RH_DIMS # = 201

def extract_landmarks(video_path, holistic):
    """
    Trích xuất các điểm mốc từ một video và trả về một mảng numpy.
    Mỗi frame sẽ là một hàng trong mảng.
    Vector đặc trưng cho mỗi frame là 201 chiều.
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
        # Chỉ lấy x, y, z cho 25 landmarks đầu tiên của pose
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:NUM_POSE_LANDMARKS]]).flatten() if results.pose_landmarks else np.zeros(POSE_DIMS)
        # Lấy x, y, z cho hand landmarks
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(LH_DIMS)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(RH_DIMS)
        
        # Tổng cộng: 201 giá trị cho mỗi frame
        frame_landmarks = np.concatenate([pose, lh, rh])
        landmarks_list.append(frame_landmarks)

    cap.release()
    return np.array(landmarks_list)

def interpolate_sequence(sequence, target_len):
    """
    Nội suy một chuỗi keypoints về một độ dài cố định.
    Args:
        sequence (np.array): Mảng numpy có shape (số_frame_gốc, 201).
        target_len (int): Độ dài mục tiêu (ví dụ: 60).
    Returns:
        np.array: Mảng numpy có shape (target_len, 201).
    """
    if sequence.shape[0] == 0:
        return np.zeros((target_len, sequence.shape[1]))
    
    # cv2.resize có thể được dùng để nội suy 1D nếu ta xem chuỗi là ảnh 1 cột
    # (height, width) -> (số_frame, số_chiều)
    interpolated_sequence = cv2.resize(sequence, (sequence.shape[1], target_len), interpolation=cv2.INTER_LINEAR)
    return interpolated_sequence

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
            # Lấy tất cả các video, không lọc
            all_videos = list(reader)
            # Lấy danh sách các video đã được xử lý
            processed_videos = {os.path.splitext(f)[0] for f in os.listdir(PROCESSED_DIR)}
            
            # Lọc ra những video chưa được xử lý
            video_files_to_process = [
                item for item in all_videos 
                if os.path.splitext(item['filename'])[0] not in processed_videos
            ]

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {LABELS_FILE}. Hãy chắc chắn file downloader.py đã chạy thành công.")
        return

    if not video_files_to_process:
        print("Tất cả video đã được xử lý. Không có gì để làm.")
        holistic.close()
        return

    print(f"Bắt đầu xử lý và trích xuất đặc trưng cho {len(video_files_to_process)} video mới...")

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for item in tqdm(video_files_to_process, desc="Đang xử lý video"):
        video_filename = item['filename']
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # Tên file output (ví dụ: D0001.mp4 -> D0001.npy)
        npy_filename = os.path.splitext(video_filename)[0] + ".npy"
        npy_path = os.path.join(PROCESSED_DIR, npy_filename)

        # Trích xuất đặc trưng
        landmarks = extract_landmarks(video_path, holistic)

        if landmarks is not None and landmarks.shape[0] > 0:
            # Chuẩn hóa độ dài chuỗi
            landmarks_interpolated = interpolate_sequence(landmarks, TARGET_FRAMES)
            
            # Lưu lại dưới dạng file numpy
            np.save(npy_path, landmarks_interpolated)

    holistic.close()
    print(f"\nHoàn tất xử lý {len(video_files_to_process)} video!")
    print(f"Dữ liệu đã được xử lý và lưu tại thư mục '{PROCESSED_DIR}'.")
    print("Bước tiếp theo là [Bước 2 - Tăng cường dữ liệu].")


if __name__ == "__main__":
    # Trước khi chạy, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết:
    # pip install opencv-python mediapipe numpy
    main()
