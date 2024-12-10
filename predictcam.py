import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("D:/CEOBosch/trainModel/runs/detect/train2/weights/best.pt")

# Open the camera (0 is the default camera ID)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predict objects in the frame
    results = model.predict(source=frame, show=True)  # `show=True` to visualize predictions directly

    # You can process results here if needed
    for result in results:
        print(result.boxes)  # Print detected boxes (if any)

    # Display the resulting frame (already displayed using show=True in YOLO)
    # Optionally, you can draw the results manually and display with OpenCV's imshow
    # cv2.imshow("YOLO Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
