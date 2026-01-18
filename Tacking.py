from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.track("GX010629.MP4", show=True)  # Tracking with default tracker