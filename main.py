import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import time

st.set_page_config(page_title="VIET SIGN LIVE", layout="centered", page_icon="🐋")

# --- Custom CSS for High-Contrast Modern Dark UI with Premium Fonts & HUD ---
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Montserrat:wght@300;500;800&family=Space+Grotesk:wght@300;500;700&display=swap');
    
    /* GLOBAL THEME OVERRIDES */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF;
    }
    
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 50% 0%, #1a1c2c 0%, #0E1117 80%);
    }

    /* HEADER STYLE */
    .header-container {
        text-align: center;
        padding: 3.5rem 1rem;
        background: linear-gradient(135deg, #00F260 0%, #0575E6 100%);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 35px rgba(0, 242, 96, 0.2);
    }
    .header-title {
        font-family: 'Orbitron', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 3.2rem;
        margin: 0;
        letter-spacing: 4px;
        text-transform: uppercase;
        text-shadow: 3px 3px 0px rgba(0,0,0,0.2);
    }
    .header-subtitle {
        font-family: 'Montserrat', sans-serif;
        color: #FFFFFF;
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-weight: 300;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    
    /* SIDEBAR STYLE */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .sidebar-header {
        font-family: 'Montserrat', sans-serif;
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        padding-left: 15px;
        border-left: 4px solid #00F260;
        margin-bottom: 2rem;
        text-transform: uppercase;
    }
    
    /* TOGGLE SWITCH STYLE (Modernizing Checkboxes) */
    .stToggle label p {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.9rem !important;
        color: #E0E0E0 !important;
        letter-spacing: 1px;
    }
    
    /* SLIDER STYLE */
    .stSlider label p {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.9rem !important;
        color: #00F260 !important;
        letter-spacing: 1px;
    }

    /* BUTTON STYLE */
    div.stButton > button {
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(90deg, #00F260 0%, #0575E6 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 5px 15px rgba(5, 117, 230, 0.3);
        width: 100%;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 10px 25px rgba(5, 117, 230, 0.5);
        background: linear-gradient(90deg, #0575E6 0%, #00F260 100%);
    }

    /* TECH TEXT STYLE (For instructions) */
    .tech-instruction {
        font-family: 'Space Grotesk', sans-serif;
        color: #A0C4FF;
        font-size: 1rem;
        background: rgba(5, 117, 230, 0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 3px solid #0575E6;
        margin-bottom: 15px;
    }
    .tech-strong {
        color: #00F260;
        font-weight: 700;
    }

    /* FUTURISTIC HUD RESULT CARD */
    .hud-card {
        background: rgba(14, 17, 23, 0.8);
        border: 1px solid #30363D;
        border-top: 4px solid #00F260;
        border-bottom: 4px solid #0575E6;
        padding: 2rem;
        border-radius: 16px;
        position: relative;
        box-shadow: 0 0 30px rgba(0, 242, 96, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        text-align: center;
        overflow: hidden;
    }
    /* Corner Accents */
    .hud-card::before {
        content: '';
        position: absolute;
        top: 10px; left: 10px;
        width: 20px; height: 20px;
        border-top: 2px solid #fff;
        border-left: 2px solid #fff;
        opacity: 0.5;
    }
    .hud-card::after {
        content: '';
        position: absolute;
        bottom: 10px; right: 10px;
        width: 20px; height: 20px;
        border-bottom: 2px solid #fff;
        border-right: 2px solid #fff;
        opacity: 0.5;
    }

    .prediction-label {
        font-family: 'Orbitron', sans-serif;
        color: #8b949e;
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .prediction-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #fff;
        text-transform: uppercase;
        margin: 0.5rem 0;
        text-shadow: 0 0 20px rgba(0, 242, 96, 0.5);
        background: -webkit-linear-gradient(#fff 40%, #a0c4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .confidence-container {
        margin-top: 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }
    .confidence-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #00F260;
    }
    .confidence-bar-bg {
        width: 80%;
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00F260, #0575E6);
        border-radius: 4px;
        box-shadow: 0 0 10px #00F260;
    }

    /* TABS STYLE */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-family: 'Montserrat', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .stTabs [aria-selected="true"] {
        color: #00F260 !important;
        border-bottom-color: #00F260 !important;
    }
    
    /* Author Info Box */
    .author-info {
        padding: 1.2rem;
        background: linear-gradient(180deg, rgba(30, 30, 30, 0.5) 0%, rgba(10, 10, 10, 0.5) 100%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    .author-label {
        font-family: 'Orbitron', sans-serif;
        color: #888;
        font-size: 0.7rem;
        letter-spacing: 2px;
        margin-bottom: 4px;
        display: block;
    }
    .author-val {
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.95rem;
        font-family: 'Space Grotesk', sans-serif;
        display: block;
        margin-bottom: 12px;
    }

</style>
""", unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">VIET SIGN LIVE</h1>
        <p class="header-subtitle">ADVANCED SIGN LANGUAGE RECOGNITION</p>
    </div>
""", unsafe_allow_html=True)

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
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 242, 96), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(5, 117, 230), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(5, 117, 230), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
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

def normalize_sequence(keypoints_sequence):
    """
    Chuẩn hóa chuỗi keypoints về khoảng [0, 1] dựa trên bounding box của cơ thể.
    Giúp model nhận diện tốt hơn bất kể khoảng cách đứng xa/gần.
    """
    normalized_sequence = []
    if not keypoints_sequence:
        return normalized_sequence

    for frame_flat in keypoints_sequence:
        if frame_flat is None:
            normalized_sequence.append(None)
            continue
        
        try:
            # Reshape về (201, 3) để xử lý
            points = frame_flat.copy().reshape(-1, 3)
            
            # Chỉ lấy các điểm có tọa độ khác 0 (valid points)
            valid_mask = np.any(points[:, :2] != 0, axis=1)
            
            if np.any(valid_mask):
                x_coords = points[valid_mask, 0]
                y_coords = points[valid_mask, 1]
                
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                
                # Chuẩn hóa X
                if (max_x - min_x) > 1e-7:
                    points[valid_mask, 0] = (x_coords - min_x) / (max_x - min_x)
                elif x_coords.size > 0:
                    points[valid_mask, 0] = 0.5 # Fallback nếu chỉ có 1 điểm hoặc thẳng hàng
                    
                # Chuẩn hóa Y
                if (max_y - min_y) > 1e-7:
                    points[valid_mask, 1] = (y_coords - min_y) / (max_y - min_y)
                elif y_coords.size > 0:
                    points[valid_mask, 1] = 0.5

            normalized_sequence.append(points.flatten())
            
        except Exception:
            normalized_sequence.append(frame_flat.copy()) # Giữ nguyên nếu lỗi

    return normalized_sequence

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

def process_webcam_to_sequence(duration_seconds, show_landmarks, holistic):
    # Đặt độ phân giải 640x480 (4:3) để kiểm tra tính ổn định
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    sequence = []
    stframe = st.empty()
    status_slot = st.empty()
    
    try:
        status_slot.warning("⏳ SYSTEM INITIALIZING...")
        while not cap.isOpened():
            time.sleep(0.1)

        ret, frame = cap.read()
        if not ret:
            st.error("CAMERA ERROR")
            return None
        
        stframe.image(cv2.flip(frame, 1), channels="BGR", width="stretch")
        status_slot.success("CAMERA ONLINE")
        time.sleep(0.5)

        countdown_start = time.time()
        countdown_duration = 3
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            elapsed_countdown = time.time() - countdown_start
            remaining = countdown_duration - int(elapsed_countdown)
            
            if remaining <= 0:
                break
            
            # Không lật frame đầu vào
            image, results = mediapipe_detection(frame, holistic)
            
            if show_landmarks:
                draw_styled_landmarks(image, results)
            
            # Lật ảnh để hiển thị
            image = cv2.flip(image, 1)

            # Vẽ số đếm ngược (đã căn giữa cho 640x480)
            cv2.putText(image, f"{remaining}", (280, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 242, 96), 15, cv2.LINE_AA)
            
            stframe.image(image, channels="BGR", width="stretch")
            status_slot.warning(f"⏳ STANDBY... {remaining}")

        start_time = time.time()
        progress = st.progress(0)
        frame_count = 0 # Biến đếm frame
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            elapsed_time = time.time() - start_time
            if elapsed_time > duration_seconds:
                break
            
            # QUAN TRỌNG: Không lật frame trước khi đưa vào MediaPipe
            # Model cần dữ liệu gốc (Raw) để dự đoán chính xác
            image, results = mediapipe_detection(frame, holistic)
            
            # KỸ THUẬT FRAME SKIPPING: Lấy 1 bỏ 1 (Giảm nhiễu & Khớp mật độ training)
            frame_count += 1
            if frame_count % 2 == 0:
                keypoints = extract_keypoints(results)
                if keypoints is not None:
                    sequence.append(keypoints)

            if show_landmarks:
                draw_styled_landmarks(image, results)
            
            # Chỉ lật hình ảnh KHI HIỂN THỊ (Hiệu ứng gương cho người dùng)
            image = cv2.flip(image, 1)

            cv2.circle(image, (image.shape[1] - 50, 50), 15, (0, 0, 255), -1) # Chỉnh lại vị trí chấm đỏ sau khi lật
            cv2.putText(image, "REC", (image.shape[1] - 130, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            stframe.image(image, channels="BGR", width="stretch")
            progress.progress(min(elapsed_time / duration_seconds, 1.0))
            status_slot.write(f"🔴 RECORDING... {elapsed_time:.1f}/{duration_seconds}s")
            
    finally:
        cap.release()
        try:
            progress.empty()
        except:
            pass
    
    return sequence

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">CẤU HÌNH HỆ THỐNG</div>', unsafe_allow_html=True)
    
    # Giảm thời gian mặc định xuống 3s để tránh khoảng thừa (silence)
    duration_seconds = st.slider("THỜI LƯỢNG GHI (S)", 2, 10, 3)
    top_k = st.slider("SỐ LƯỢNG KẾT QUẢ", 1, 5, 1)
    
    st.markdown("---")
    
    show_landmarks = st.toggle("HIỂN THỊ LANDMARKS", value=True)
    debug_mode = st.toggle("CHẾ ĐỘ GỠ LỖI", value=False)

    st.markdown("---")
    
    st.markdown(f"""
    <div class="author-info">
        <span class="author-label">DEVELOPED BY</span>
        <span class="author-val">Tung Lam Nguyen</span>
        <span class="author-label">STUDENT ID</span>
        <span class="author-val">23IT138</span>
    </div>
    """, unsafe_allow_html=True)

# --- Main UI ---
tab_cam, tab_video = st.tabs(["CAMERA TRỰC TIẾP", "TẢI LÊN VIDEO"])

sequence = None
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with tab_cam:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div class="tech-instruction">
            ⏱️ Thời lượng mặc định: <span class="tech-strong">{duration_seconds} giây</span>. 
            Nhấn nút bên cạnh để bắt đầu.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        start_btn = st.button("KÍCH HOẠT CAMERA", type="primary", use_container_width=True)
    
    if start_btn:
        sequence = process_webcam_to_sequence(duration_seconds, show_landmarks, holistic)

with tab_video:
    st.markdown('<div class="tech-instruction">Hỗ trợ định dạng <span class="tech-strong">.MP4, .AVI</span>. Kéo thả file vào khung bên dưới.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD FILE", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        st.video(tmp_path)
        if st.button("PHÂN TÍCH VIDEO", type="primary", use_container_width=True):
            with st.spinner("PROCESSING..."):
                sequence = sequence_frames(tmp_path, holistic)

# Dự đoán
if sequence is not None:
    st.markdown("---")
    
    # BƯỚC QUAN TRỌNG: Chuẩn hóa dữ liệu đầu vào để khớp với training
    norm_sequence = normalize_sequence(sequence)
    
    # Sau đó mới nội suy về 60 frames
    kp = interpolate_keypoints(norm_sequence)
    
    if kp is not None: # Kiểm tra an toàn
        # FIX QUAN TRỌNG: Kẹp giá trị (Clip) về [0, 1]
        kp = np.clip(kp, 0.0, 1.0)

        # Chế độ chạy thật bằng Model
        result = model.predict(np.expand_dims(kp, axis=0))[0] 
        top_indices = np.argsort(result)[-top_k:][::-1] 
        top_probs = result[top_indices]
        top_labels = [inv_label_map[i] for i in top_indices]

        # Hiển thị kết quả
        col_res, col_chart = st.columns([1, 1])

        with col_res:
            st.markdown(f"""
            <div class="hud-card">
                <div class="prediction-label">PREDICTION RESULT</div>
                <div class="prediction-text">{top_labels[0]}</div>
                <div class="confidence-container">
                    <div class="confidence-value">{top_probs[0]:.1%}</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {top_probs[0]*100}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            if top_k > 1:
                st.markdown("#### PROBABILITY CHART")
                chart_data = dict(zip(top_labels, top_probs))
                st.bar_chart(chart_data, color="#00F260")
            else:
                st.info("💡 Tăng 'SỐ LƯỢNG KẾT QUẢ' trong cấu hình để xem chi tiết hơn.")
    
    if debug_mode:
        with st.expander("🛠️ SYSTEM LOGS"):
            st.json(dict(zip(top_labels, top_probs)))
    else:
        pass