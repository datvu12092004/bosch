import cv2
from ultralytics import YOLO
import time
import winsound  # Thư viện để phát âm thanh trên Windows

# Load the pre-trained YOLO model
model = YOLO("D:/CEOBosch/trainModel/runs/detect/train2/weights/best.pt")

# Open the camera (0 is the default camera ID)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Biến kiểm tra trạng thái và thời gian
start_time = None
MICROSLEEP_THRESHOLD = 2  # Thời gian ngưỡng cho microsleep (giây)
alarm_on = False  # Biến trạng thái cảnh báo

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predict objects in the frame
    results = model.predict(source=frame, conf=0.5, show=True)  # Adjust confidence threshold if needed

    # Kiểm tra trạng thái microsleep
    microsleep_in_frame = False
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Giả định nhãn 0 là "Closed" (cần đảm bảo khớp với mô hình của bạn)
                microsleep_in_frame = True
                break

    if microsleep_in_frame:
        if start_time is None:  # Bắt đầu đếm thời gian nếu phát hiện microsleep
            start_time = time.time()
        elif time.time() - start_time >= MICROSLEEP_THRESHOLD and not alarm_on:
            # Sau 2 giây liên tục microsleep, bật cảnh báo
            alarm_on = True
            print("YOU ARE IN MICROSLEEP")
    else:
        # Reset nếu phát hiện trạng thái focus
        if alarm_on:  # Chỉ tắt cảnh báo nếu đang bật
            print("Focus detected, stopping alarm.")
        alarm_on = False
        start_time = None

    # Hiển thị cảnh báo trên khung hình nếu cần
    if alarm_on:
        winsound.Beep(1000, 500)  # Phát âm thanh cảnh báo
        cv2.putText(frame, "YOU ARE IN MICROSLEEP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị frame
    cv2.imshow("YOLO Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
