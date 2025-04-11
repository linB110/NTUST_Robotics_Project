from ultralytics import YOLO
import cv2

# This python test code is used for AABB model + img input

model = YOLO('Qrcode.pt')   # change to your model

class_names = model.names

# loading imgae
image_path = 'barcode.jpg'  # your image path
image = cv2.imread(image_path)

results = model(image)

for result in results:
    boxes = result.boxes  # detected frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get frame cordinate
        confidence = box.conf[0]  # get confidenct
        class_id = int(box.cls[0])  # get class ID

        # tranfer ID to Name 
        class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'

        # plotting frame and cordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 0), 2)

# show result
cv2.imshow('YOLOv8 Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
