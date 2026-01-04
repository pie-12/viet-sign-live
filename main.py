import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import time

st.set_page_config(page_title="VSL Prediction", layout="centered")

# --- Hero Header ---
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #008080;">VSL RECOGNITION SYSTEM</h1>
        <p style="color: gray; font-size: 1.2em;">Hệ thống nhận diện Ngôn ngữ ký hiệu Việt Nam</p>
        <code style="font-size: 0.8em;">Input: 60x201 | Model: BiLSTM | Labels: 2764</code>
    </div>
""", unsafe_allow_html=True)
st.divider()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = []
for connection in ALL_POSE_CONNECTIONS:
    if connection[0] < N_UPPER_BODY_POSE_LANDMARKS and connection[1] < N_UPPER_BODY_POSE_LANDMARKS:
        UPPER_BODY_POSE_CONNECTIONS.append(connection)

# ====================
# Load model và label_map
# ====================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Models/checkpoints/final_model.keras')

@st.cache_data
def load_label_map():
    import json
    with open('Logs/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

model = load_model()
label_map, inv_label_map = load_label_map()

# ====================
# Hàm xử lý video
# ====================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Hands
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    keypoints = np.concatenate([pose_kps,left_hand_kps, right_hand_kps])
    return keypoints.flatten()

def interpolate_keypoints(keypoints_sequence, target_len = 60):
    if len(keypoints_sequence) == 0:
        return None
    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))
    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        interpolator = interp1d(original_times, feature_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
        interpolated_sequence[:, feature_idx] = interpolator(target_times)
    return interpolated_sequence

def sequence_frames(video_path, holistic):
  sequence_frames = []
  cap = cv2.VideoCapture(video_path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  step = max(1, total_frames // 100)
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret: break
      if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0: continue
      try:
          image, results = mediapipe_detection(frame, holistic)
          keypoints = extract_keypoints(results)
          if keypoints is not None: sequence_frames.append(keypoints)
      except Exception as e: continue
  cap.release()
  return sequence_frames

def process_webcam_to_sequence(duration_seconds, show_landmarks):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    sequence = []
    stframe = st.empty()
    status_slot = st.empty()
    
    # Khởi tạo model sớm
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        # --- Giai đoạn 1: Đếm ngược trực tiếp trên camera ---
        countdown_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            elapsed_countdown = time.time() - countdown_start
            remaining = 3 - int(elapsed_countdown)
            
            # Lật ảnh cho giống gương
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            
            if show_landmarks:
                draw_styled_landmarks(image, results)
            
            # Hiển thị chữ đếm ngược lên ảnh
            cv2.putText(image, f"READY: {remaining}", (150, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
            
            stframe.image(image, channels="BGR", use_container_width=True)
            status_slot.warning(f"⏳ Chuẩn bị... Bắt đầu sau {remaining} giây")
            
            if elapsed_countdown >= 3:
                break

        # --- Giai đoạn 2: Ghi hình ---
        start_time = time.time()
        progress = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            elapsed_time = time.time() - start_time
            if elapsed_time > duration_seconds:
                break
            
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            
            # Trích xuất keypoints
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence.append(keypoints)

            if show_landmarks:
                draw_styled_landmarks(image, results)
            
            # Hiển thị biểu tượng REC
            cv2.circle(image, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(image, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            stframe.image(image, channels="BGR", use_container_width=True)
            progress.progress(min(elapsed_time / duration_seconds, 1.0))
            status_slot.write(f"🔴 Đang ghi... {elapsed_time:.1f}/{duration_seconds}s")
            
    finally:
        cap.release()
        holistic.close()
        try:
            progress.empty()
        except:
            pass
    
    return sequence

# Streamlit App

# --- Sidebar ---
st.sidebar.title("⚙️ Cấu hình")
duration_seconds = st.sidebar.slider("⏱️ Thời gian ghi hình (s)", 2, 10, 4)
top_k = st.sidebar.slider("📊 Số lượng gợi ý (Top-k)", 1, 10, 1)
show_landmarks = st.sidebar.checkbox("🦴 Hiển thị Landmarks", value=True)
debug_mode = st.sidebar.checkbox("🐞 Chế độ Debug", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Hướng dẫn")
st.sidebar.markdown("""
1. Chọn chế độ **Webcam** hoặc **Video**.
2. Nhấn nút **Ghi hình** hoặc **Tải video**.
3. Xem kết quả dự đoán và độ tin cậy.
""")

# --- Main UI ---
tab_cam, tab_video = st.tabs(["📷 Webcam", "🎞️ Video file"])

sequence = None
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with tab_cam:
    st.info(f"Nhấn nút bên dưới để bắt đầu ghi hình trong {duration_seconds} giây.")
    if st.button("📸 Bắt đầu ghi hình"):
        sequence = process_webcam_to_sequence(duration_seconds, show_landmarks)

with tab_video:
    uploaded_file = st.file_uploader("Tải lên video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        if st.button("🔍 Phân tích video"):
            sequence = sequence_frames(tmp_path, holistic)

# Dự đoán
if sequence is not None:
    status_text = st.empty()
    status_text.text("⏳ Processing keypoints...")
    kp = interpolate_keypoints(sequence)
    
    status_text.text("🧠 Predicting...")
    result = model.predict(np.expand_dims(kp, axis=0))[0] 
    status_text.empty()

    # Xử lý kết quả
    top_indices = np.argsort(result)[-top_k:][::-1] 
    top_probs = result[top_indices]
    top_labels = [inv_label_map[i] for i in top_indices]

    pred_label = top_labels[0]
    confidence = top_probs[0]

    # Hiển thị kết quả (Modern Layout)
    st.divider()
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Kết quả dự đoán")
        st.markdown(f"<h1 style='color: #4CAF50;'>{pred_label}</h1>", unsafe_allow_html=True)
        st.metric("Độ tin cậy", f"{confidence:.1%}")

    with col2:
        st.subheader(f"Top {top_k} Khả năng")
        if top_k > 1:
            chart_data = dict(zip(top_labels, top_probs))
            st.bar_chart(chart_data)
        else:
            st.info("Tăng 'Top-k' trong sidebar để xem thêm các dự đoán khác.")

    if debug_mode:
        with st.expander("🛠️ Technical Details"):
            st.write(f"**Input Shape:** {kp.shape}")
            st.write(f"**Top Probs:** {top_probs}")
            st.json(dict(zip(top_labels, top_probs)))
