from ultralytics import YOLO
import numpy as np
from collections import Counter
import cv2 as cv


img = cv.imread("canopy1.jpg")
model = YOLO("runs/detect/train/weights/best.pt")

result = model.predict(img, save=True)
labels = result[0].boxes.cls.cpu().numpy()

counts = Counter(labels)

annotated_image = result[0].plot()
annotated_image = cv.resize(annotated_image, (1000,500))

y_offset = 30
font = cv.FONT_HERSHEY_SIMPLEX

for class_id, count in counts.items():
    class_name = model.names[int(class_id)]
    text = f"{class_name}: {count}"
    cv.putText(annotated_image, text, (10, y_offset), font, 0.8, (0, 0, 0), 2)
    y_offset += 30


cv.imwrite("output_with_counts.jpg", annotated_image)

# Or display it
cv.imshow("Image with Object Counts", annotated_image)
cv.waitKey(0)
cv.destroyAllWindows()