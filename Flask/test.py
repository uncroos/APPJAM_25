from ultralytics import YOLO
import cv2
from PIL.Image import open
import numpy as np

img = cv2.imread("./teest.jpg")
model_path = "./model.pt"
model = YOLO(model_path)
threshold = 0.5

results = model(img)[0]

for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
                cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6, cv2.LINE_AA)

# print(img.shape)
# img = np.transpose((2, 3, 1)).squeeze()
img = np.array(img, dtype=np.uint8)
img = cv2.resize(img, dsize=(640, 640))
print(img.shape)

while True:
    cv2.imshow("img", img)
    if cv2.waitKey(0) == ord("q"):
        break

cv2.destroyAllWindows()