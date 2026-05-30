from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("best.pt")
print(f"Model loaded | Classes: {model.names}")

CONF_THRESHOLD = 0.45

CLASS_COLORS = {
    0: (255, 100,  50),  # plastic_bottle
    1: ( 50, 230,  50),  # backpack
    2: ( 50, 150, 255),  # perfume
    3: (  0, 255, 255),  # clock
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_time = time.time()
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        color = CLASS_COLORS.get(cls_id, (200,200,200))

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1,y1-lh-8), (x1+lw,y1), color, -1)
        cv2.putText(frame, label, (x1,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        print(f"  → {class_name} | conf={conf:.2f}")

    fps = 1.0 / (time.time() - prev_time + 1e-6)
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("YOLOv8 Detection - Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
