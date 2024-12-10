from ultralytics import YOLO
import cv2
import time
import winsound  # Để phát âm thanh cảnh báo (trên Windows)

# Load the pre-trained YOLO model
model = YOLO("D:/CEOBosch/trainModel/runs/detect/train2/weights/best.pt")

# Open a video file or webcam feed
video_path = "D:/CEOBosch/trainModel/mypaperdataset.mp4"  # Đường dẫn tới video
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu mở được video
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Lấy kích thước màn hình laptop (ví dụ: 1920x1080)
screen_width = 1920
screen_height = 1080

# Lấy kích thước gốc của video
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tính toán tỉ lệ sao cho kích thước video không vượt quá màn hình
width_ratio = screen_width / video_width
height_ratio = screen_height / video_height
resize_ratio = min(width_ratio, height_ratio)

# Biến để theo dõi trạng thái "microsleep"
microsleep_detected = False
microsleep_start_time = None
focus_detected = False

# Đọc từng frame trong video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Kết thúc video

    # Dự đoán trên từng frame
    results = model.predict(frame, imgsz=640, conf=0.25)  # Bạn có thể tùy chỉnh imgsz và conf

    # Vẽ kết quả lên frame
    annotated_frame = results[0].plot()  # Annotated frame với bounding boxes

    # Kiểm tra trạng thái "microsleep"
    for result in results[0].boxes.data:
        class_id = int(result[5])  # Lấy ID lớp của đối tượng (ví dụ: "Closed" cho mắt đóng)
        if class_id == 0:  # Giả sử "0" là mã cho mắt đóng (closed)
            if not microsleep_detected:
                microsleep_start_time = time.time()
                microsleep_detected = True
        else:
            if microsleep_detected:
                microsleep_detected = False
                microsleep_start_time = None

    # Kiểm tra nếu microsleep kéo dài 2 giây liên tiếp
    if microsleep_detected and time.time() - microsleep_start_time >= 2:
        # Hiển thị thông báo cảnh báo
        cv2.putText(annotated_frame, "You are in Microsleep!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Phát âm thanh cảnh báo
        winsound.Beep(1000, 500)  # Tần số 1000 Hz, kéo dài 500 ms
    else:
        # Nếu có "focus" (mắt mở), tắt cảnh báo
        if not microsleep_detected and focus_detected:
            cv2.putText(annotated_frame, "Focus detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Thay đổi kích thước frame sao cho không vượt quá màn hình laptop
    new_width = int(video_width * resize_ratio)
    new_height = int(video_height * resize_ratio)
    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

    # Hiển thị frame đã được xử lý
    cv2.imshow("YOLO Detection", resized_frame)

    # Dừng video khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
