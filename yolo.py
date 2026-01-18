from ultralytics import YOLO
import numpy as np


if __name__ == '__main__':

    model = YOLO("yolo11n.pt")
    train = model.train(data='Data/data.yaml', epochs=100, imgsz=640, workers=1)
    metrics = model.val()
    results = model("leaf_1.jpg")
    print(results)

