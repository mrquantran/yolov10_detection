import os
from ultralytics import YOLO

# Define paths
MODEL_PATH = "./yolov10n.pt"
YAML_PATH = "./datasets/data.yaml"
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 256
TRAINED_MODEL_PATH = 'runs/detect/train/weights/best.pt'

# Verify paths


def check_path(path):
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return False
    return True


# Check if the model path exists
if not check_path(MODEL_PATH):
    raise FileNotFoundError(f"Model path {MODEL_PATH} not found.")

# Check if the YAML path exists
if not check_path(YAML_PATH):
    raise FileNotFoundError(f"YAML path {YAML_PATH} not found.")

# Load model
model = YOLO(MODEL_PATH)

# Train the model
try:
    model.train(data=YAML_PATH,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                save_dir='runs/detect/train'
                )
except Exception as e:
    print(f"An error occurred during training: {e}")

model_trained = YOLO(TRAINED_MODEL_PATH)
# Validate the model
try:
    model_trained.val(data=YAML_PATH,
                      imgsz=IMG_SIZE,
                      split='test')
except Exception as e:
    print(f"An error occurred during validation: {e}")
