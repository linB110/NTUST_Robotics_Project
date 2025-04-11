from ultralytics import YOLO
import cv2
import math

# This python test code is used for OBB model + camera input

model = YOLO('obbBox.onnx', task='obb')

# intialize camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    # inferencing
    results = model.predict(frame)

    # plotting result on screen
    annotated_frame = results[0].plot()

    # show screen
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # filtering
    for result in results:
        if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
            obb_boxes = result.obb.xywhr  # return OBB cordinate
            scores = result.obb.conf      # confidence
            classes = result.obb.cls      # class ID

            for box, score, cls in zip(obb_boxes, scores, classes):

                # check if class ID is in Name dictionary
                if int(cls) in model.names:
                    print(f"ClassID: {model.names[int(cls)]}, Conficence: {score}, Frame: {box}")
                else:
                    print(f"Unknown ID: {cls}, Conficence: {score}, Frame: {box}")
        else:
            print("Not detected")


    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and quit
cap.release()
cv2.destroyAllWindows()
