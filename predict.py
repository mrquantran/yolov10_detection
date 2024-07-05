from ultralytics import YOLO
import cv2
TRAINED_MODEL_PATH = 'runs/detect/train/weights/best.pt'
model = YOLO(TRAINED_MODEL_PATH)

IMAGE_URL = 'https://ips-dc.org/wp-content/uploads/2022/05/Black-Workers-Need-a-Bill-of-Rights.jpeg'
CONF_THRESHOLD = 0.3
results = model.predict(source=IMAGE_URL,
                        imgsz=640,
                        conf=CONF_THRESHOLD)
annotated_img = results[0].plot()

cv2.imshow('', annotated_img)
cv2.waitKey(0)
