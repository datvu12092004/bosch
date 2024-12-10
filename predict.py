from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolov8s.pt")
model = YOLO("D:/CEOBosch/trainModel/runs/detect/train2/weights/best.pt")
# Start training on your custom dataset
results = model.predict("D:/BOSCH/Sub2/Eyeclose/1103.jpg")
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()
    result.save(filename="result.jpg")