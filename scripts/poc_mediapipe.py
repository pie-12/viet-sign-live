import cv2
import mediapipe as mp

# Khởi tạo các module của MediaPipe
mp_drawing = mp.solutions.drawing_utils       # Công cụ để vẽ landmarks
mp_drawing_styles = mp.solutions.drawing_styles # Phong cách vẽ
mp_holistic = mp.solutions.holistic           # Module Holistic (tất cả trong một)

print("Đang khởi động webcam... (Bấm 'q' để thoát)")
cap = cv2.VideoCapture(0) # Số 0 là webcam mặc định

# Khởi tạo mô hình Holistic
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        # Đọc một khung hình từ webcam
        success, image = cap.read()
        if not success:
            print("Không thể đọc frame từ webcam. Bỏ qua...")
            continue

        # Lật ảnh theo chiều ngang để có góc nhìn "gương soi"
        image = cv2.flip(image, 1)

        # Chuyển màu từ BGR (OpenCV) sang RGB (MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Xử lý ảnh và phát hiện landmarks
        results = holistic.process(image_rgb)

        # Vẽ landmarks lên ảnh gốc (BGR)
        
        # 1. Vẽ landmarks khuôn mặt
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # 2. Vẽ landmarks cơ thể (pose)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # 3. Vẽ landmarks bàn tay TRÁI
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

        # 4. Vẽ landmarks bàn tay PHẢI
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

        # Hiển thị ảnh kết quả
        cv2.imshow('Viet Sign Live - POC MediaPipe (Bấm q để thoát)', image)

        # Bấm 'q' để thoát
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
print("Đã đóng webcam.")