import requests
import os
import csv
from tqdm import tqdm
import time

# --- Cấu hình ---
API_URL = "https://qipedc.moet.gov.vn/dictionary/getAll"
BASE_VIDEO_URL = "https://qipedc.moet.gov.vn/videos/"
DATA_DIR = r"D:\Dataset\VietSignLive"
VIDEO_DIR = os.path.join(DATA_DIR, "videos_mp4")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
ITEMS_PER_PAGE = 100 # Lấy 100 mục mỗi lần để giảm số lượng request

def download_video(video_url, file_path):
    """
    Tải một file video một cách an toàn, sử dụng file tạm (.part) 
    để tránh tạo ra file hỏng nếu quá trình tải bị ngắt quãng.
    """
    temp_file_path = file_path + ".part"
    try:
        response = requests.get(video_url, stream=True, verify=False)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(temp_file_path, 'wb') as f, tqdm(
            desc=os.path.basename(file_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False # Không để lại thanh tiến trình của file sau khi hoàn tất
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        # Đổi tên file tạm thành file chính thức sau khi tải xong
        os.rename(temp_file_path, file_path)
        return True

    except requests.exceptions.RequestException as e:
        tqdm.write(f"Lỗi khi tải {video_url}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False
    except KeyboardInterrupt:
        tqdm.write("\nNgười dùng đã dừng. Đang xóa file tải dở...")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Ném lại lỗi để chương trình chính dừng lại
        raise
    except Exception as e:
        tqdm.write(f"Lỗi không xác định khi tải {video_url}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False

def main():
    """
    Hàm chính để lấy dữ liệu từ API, tải video và tạo file labels.
    """
    print("Bắt đầu quá trình tải dữ liệu...")

    # Tạo các thư mục nếu chưa tồn tại
    os.makedirs(VIDEO_DIR, exist_ok=True)
    print(f"Thư mục video: {VIDEO_DIR}")
    print(f"File labels: {LABELS_FILE}")

    all_signs = []
    current_page = 1
    total_pages = 1 # Giả định ban đầu là có 1 trang, sẽ cập nhật sau

    # Vòng lặp để xử lý phân trang (pagination)
    while current_page <= total_pages:
        print(f"\nĐang lấy dữ liệu từ trang {current_page}/{total_pages}...")
        payload = {
            'group': ITEMS_PER_PAGE,
            'text': '',
            'page': current_page
        }
        
        try:
            response = requests.post(API_URL, data=payload, verify=False)
            response.raise_for_status()
            data = response.json()

            # --- PHẦN QUAN TRỌNG: Phân tích cấu trúc JSON ---
            # Giả định rằng JSON trả về có một key là 'data' chứa danh sách các mục
            # và một key khác chứa thông tin về tổng số trang, ví dụ: 'totalPages'
            if not data.get('data'):
                 print("Lỗi: Không tìm thấy key 'data' trong phản hồi JSON. Vui lòng kiểm tra lại cấu trúc JSON.")
                 print("Nội dung JSON nhận được:", data)
                 break

            signs_on_page = data['data']
            all_signs.extend(signs_on_page)

            # Cập nhật tổng số trang từ phản hồi của API (nếu có)
            # Đây là một giả định, tên key 'totalPages' có thể khác
            if 'totalPages' in data and total_pages == 1:
                total_pages = data['totalPages']
                print(f"Phát hiện tổng số trang là: {total_pages}")

            print(f"Đã lấy thành công {len(signs_on_page)} mục từ trang {current_page}.")
            current_page += 1
            
            # Thêm một khoảng nghỉ nhỏ để tránh gửi quá nhiều request cùng lúc
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi gọi API ở trang {current_page}: {e}")
            break
        except ValueError: # Bắt lỗi JSONDecodeError
            print("Lỗi: Không thể giải mã JSON từ phản hồi. Có thể API đã trả về lỗi HTML.")
            print("Nội dung phản hồi:", response.text)
            break

    if not all_signs:
        print("Không lấy được dữ liệu nào. Dừng chương trình.")
        return

    print(f"\nTổng cộng đã lấy được thông tin của {len(all_signs)} ký hiệu.")
    print("Bắt đầu tải video và tạo file labels.csv...")

    # --- LOGIC ĐỂ TIẾP TỤC ---
    processed_files = set()
    try:
        if os.path.exists(LABELS_FILE) and os.path.getsize(LABELS_FILE) > 0:
            print("Phát hiện file labels.csv đã có. Đang đọc danh sách file đã xử lý...")
            with open(LABELS_FILE, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader) # Đọc header
                if header != ['filename', 'label']:
                    print("Cảnh báo: File labels.csv có header không mong muốn. Có thể sẽ ghi lại từ đầu.")
                else:
                    for row in reader:
                        if row:
                            processed_files.add(row[0])
            print(f"Đã tìm thấy {len(processed_files)} file đã được xử lý.")
    except (IOError, StopIteration, csv.Error) as e:
        print(f"Lỗi khi đọc file labels.csv: {e}. Sẽ bắt đầu lại từ đầu.")
        processed_files = set()


    # Mở file ở chế độ 'a' (append)
    with open(LABELS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Ghi header nếu file còn trống
        if not processed_files and os.path.getsize(LABELS_FILE) == 0:
            csv_writer.writerow(['filename', 'label'])

        for sign in tqdm(all_signs, desc="Tổng tiến trình"):

            label = sign.get('word')
            code = sign.get('_id')

            if not label or not code:
                continue

            video_filename = f"{code}.mp4"

            # Bỏ qua nếu đã xử lý
            if video_filename in processed_files:
                continue

            video_filepath = os.path.join(VIDEO_DIR, video_filename)

            # Tải video nếu chưa tồn tại
            if not os.path.exists(video_filepath):
                video_url = f"{BASE_VIDEO_URL}{video_filename}"
                tqdm.write(f"\nĐang tải: {video_url}") # Sử dụng tqdm.write
                if not download_video(video_url, video_filepath):
                    tqdm.write(f"Tải video {video_filename} thất bại. Bỏ qua.")
                    continue  # Bỏ qua nếu tải lỗi

            # Ghi vào CSV sau khi chắc chắn video tồn tại (hoặc vừa tải xong)
            csv_writer.writerow([video_filename, label])
            processed_files.add(video_filename) # Cập nhật set trong phiên chạy hiện tại

    print("\nHoàn tất! Dữ liệu đã được tải và file labels.csv đã được cập nhật.")
    print(f"-> Thư mục video: {VIDEO_DIR}")
    print(f"-> File labels: {LABELS_FILE}")


if __name__ == "__main__":
    main()
