from ultralytics import YOLO
import cv2

# This python test code is used for AABB model + camera input


model = YOLO("local.onnx")

# initialize camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    results = model(frame)

    # plotting result on screen
    annotated_frame = results[0].plot()

    # show screen
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # filtering
    for result in results:
        for box in result.boxes:
            if box.conf > 0.6:  # setting confidence threshold
                print(box.xyxy)  # show if qualified
                print(box.conf)  # show confidence
                print(box.cls)   # show object type

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close window
cap.release()
cv2.destroyAllWindows()
