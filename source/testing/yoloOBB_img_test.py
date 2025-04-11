from ultralytics import YOLO

# This python test code is used for OBB model + camera input

model = YOLO('obbBox.pt')  

results = model.predict('box.jpg')  # this is your image path

# rrunning iteration
for result in results:
    result.show()  # show image and prediction

    # print to debug
    #print("Result contents:", result)

    # check if OBB is exist
    if result.obb is not None and len(result.obb) > 0:
        # get OBB frame (cx, cy, w, h, angle)
        obb_boxes = result.obb.xywhr  # return OBB cordinate

        # get confidence and class ID
        scores = result.obb.conf  # confidence
        classes = result.obb.cls  # class ID

        # go through every detected object
        for box, score, cls in zip(obb_boxes, scores, classes):
            # box format is [cx, cy, w, h, angle]
            print(f"classID: {cls}, Confidence: {score}, Frame: {box}")
    else:
        print("No detected")
