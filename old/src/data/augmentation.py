import numpy as np

# Định nghĩa các hằng số cho cấu trúc keypoint
# 25 pose landmarks (x,y,z), 21 left hand (x,y,z), 21 right hand (x,y,z)
NUM_POSE_LANDMARKS = 25
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + (2 * NUM_HAND_LANDMARKS) # 67 landmarks
COORDS_PER_LANDMARK = 3 # x, y, z

POSE_DIMS = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK # 75
LH_DIMS = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK # 63
RH_DIMS = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK # 63
FRAME_DIM = POSE_DIMS + LH_DIMS + RH_DIMS # 201

# Chú ý: Các hàm augmentation sẽ làm việc trên dữ liệu có shape (T, 201)
# Trong đó T là số frames (thường là 60 sau preprocessing), 201 là số chiều đặc trưng

def rotate_keypoints(sequence, max_angle=10):
    """
    Xoay các keypoints 3D quanh trục Z (trong mặt phẳng X-Y).
    Args:
        sequence (np.array): Chuỗi keypoints (T, 201).
        max_angle (float): Góc xoay tối đa (độ).
    Returns:
        np.array: Chuỗi keypoints đã xoay.
    """
    rotated_sequence = sequence.copy()
    
    # Tạo một góc xoay ngẫu nhiên
    angle = np.random.uniform(-max_angle, max_angle)
    rad = np.deg2rad(angle)
    
    # Ma trận xoay quanh trục Z
    Rz = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]
    ])
    
    # Áp dụng ma trận xoay cho từng keypoint trong từng frame
    for t in range(sequence.shape[0]): # Duyệt qua từng frame
        for i in range(TOTAL_LANDMARKS): # Duyệt qua từng landmark
            # Lấy tọa độ (x, y, z) của landmark
            start_idx = i * COORDS_PER_LANDMARK
            end_idx = start_idx + COORDS_PER_LANDMARK
            point = sequence[t, start_idx:end_idx]
            
            # Nếu point không phải là vector 3D hợp lệ (ví dụ: là phần đệm 0), bỏ qua
            if np.all(point == 0):
                continue

            # Áp dụng xoay
            rotated_point = Rz @ point
            rotated_sequence[t, start_idx:end_idx] = rotated_point
            
    return rotated_sequence

def translate_keypoints(sequence, max_translation_factor=0.05):
    """
    Dịch chuyển toàn bộ keypoints trong một frame theo một vector ngẫu nhiên.
    Args:
        sequence (np.array): Chuỗi keypoints (T, 201).
        max_translation_factor (float): Tỷ lệ dịch chuyển tối đa so với range [0,1].
                                        Ví dụ, 0.05 nghĩa là dịch chuyển tối đa 5% chiều rộng/cao của frame ảo.
    Returns:
        np.array: Chuỗi keypoints đã dịch chuyển.
    """
    translated_sequence = sequence.copy()
    
    # Tạo vector dịch chuyển ngẫu nhiên (tx, ty, tz)
    # Giả định tọa độ x, y, z nằm trong khoảng [0, 1] hoặc tương tự
    tx = np.random.uniform(-max_translation_factor, max_translation_factor)
    ty = np.random.uniform(-max_translation_factor, max_translation_factor)
    tz = np.random.uniform(-max_translation_factor, max_translation_factor) # Dịch chuyển trên trục Z cũng có thể hữu ích
    
    translation_vector = np.array([tx, ty, tz])
    
    # Áp dụng dịch chuyển cho từng keypoint trong từng frame
    for t in range(sequence.shape[0]): # Duyệt qua từng frame
        for i in range(TOTAL_LANDMARKS): # Duyệt qua từng landmark
            start_idx = i * COORDS_PER_LANDMARK
            end_idx = start_idx + COORDS_PER_LANDMARK
            
            # Nếu point không phải là vector 3D hợp lệ (ví dụ: là phần đệm 0), bỏ qua
            if np.all(translated_sequence[t, start_idx:end_idx] == 0):
                continue
            
            translated_sequence[t, start_idx:end_idx] += translation_vector
            
    return translated_sequence

def scale_keypoints(sequence, max_scale_factor=0.1):
    """
    Phóng to/thu nhỏ toàn bộ keypoints.
    Args:
        sequence (np.array): Chuỗi keypoints (T, 201).
        max_scale_factor (float): Tỷ lệ thay đổi kích thước tối đa (ví dụ: 0.1 nghĩa là scale từ 0.9x đến 1.1x).
    Returns:
        np.array: Chuỗi keypoints đã được scale.
    """
    scaled_sequence = sequence.copy()
    
    # Tạo hệ số scale ngẫu nhiên
    scale = np.random.uniform(1 - max_scale_factor, 1 + max_scale_factor)
    
    # Áp dụng scale cho toàn bộ sequence (chỉ cho các tọa độ x, y, z)
    # Chúng ta phải scale từng nhóm 3 tọa độ (x,y,z) riêng biệt
    for t in range(sequence.shape[0]):
        for i in range(TOTAL_LANDMARKS):
            start_idx = i * COORDS_PER_LANDMARK
            end_idx = start_idx + COORDS_PER_LANDMARK
            
            if np.all(scaled_sequence[t, start_idx:end_idx] == 0):
                continue
                
            scaled_sequence[t, start_idx:end_idx] *= scale
            
    return scaled_sequence


def temporal_speed_variation(sequence, max_speed_change=0.2):
    """
    Thay đổi tốc độ thời gian của chuỗi keypoints bằng cách nội suy.
    Args:
        sequence (np.array): Chuỗi keypoints (T, 201), T = 60.
        max_speed_change (float): Tỷ lệ thay đổi tốc độ tối đa (ví dụ: 0.2 nghĩa là từ 0.8x đến 1.2x).
    Returns:
        np.array: Chuỗi keypoints đã thay đổi tốc độ thời gian, sau đó nội suy lại về độ dài gốc (60).
    """
    T, D = sequence.shape # T=60, D=201
    
    # Tạo hệ số thay đổi tốc độ ngẫu nhiên
    speed_factor = np.random.uniform(1 - max_speed_change, 1 + max_speed_change)
    
    # Tính toán độ dài mới
    new_T_float = T / speed_factor
    new_T = int(np.round(new_T_float)) # Làm tròn để có số frame mới
    
    # Đảm bảo độ dài mới không quá nhỏ hoặc quá lớn một cách phi lý
    if new_T < 5: new_T = 5
    if new_T > T * 2: new_T = T * 2 # Giới hạn độ dài mới để tránh lỗi hoặc quá chậm
    
    # Tạo mảng thời gian gốc và mới
    original_time = np.linspace(0, T - 1, T)
    new_time = np.linspace(0, T - 1, new_T)
    
    varied_sequence = np.zeros((new_T, D))
    
    # Nội suy từng chiều đặc trưng
    for dim in range(D):
        varied_sequence[:, dim] = np.interp(new_time, original_time, sequence[:, dim])
        
    # Sau khi thay đổi tốc độ, nội suy lại về độ dài ban đầu (T=60)
    final_sequence = np.zeros_like(sequence)
    for dim in range(D):
        final_sequence[:, dim] = np.interp(np.linspace(0, new_T - 1, T), np.linspace(0, new_T - 1, new_T), varied_sequence[:, dim])
        
    return final_sequence

def augment_sequence(sequence, num_augmentations=1):
    """
    Áp dụng một hoặc nhiều phép tăng cường cho một chuỗi keypoints.
    Args:
        sequence (np.array): Chuỗi keypoints (T, 201).
        num_augmentations (int): Số lượng biến đổi ngẫu nhiên sẽ áp dụng.
    Returns:
        np.array: Chuỗi keypoints đã được tăng cường.
    """
    augmented_sequence = sequence.copy()
    
    # Danh sách các hàm augmentation có thể áp dụng
    augmentation_funcs = [
        rotate_keypoints,
        translate_keypoints,
        scale_keypoints,
        temporal_speed_variation
        # inter_hand_distance_adjustment, # Chưa triển khai
    ]
    
    # Chọn ngẫu nhiên N phép biến đổi
    chosen_funcs = np.random.choice(augmentation_funcs, num_augmentations, replace=False)
    
    for func in chosen_funcs:
        augmented_sequence = func(augmented_sequence)
        
    return augmented_sequence

