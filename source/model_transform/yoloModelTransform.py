from ultralytics import YOLO

# load your yolov8.py model 
model = YOLO('obbBox.pt')  # change to your model.pt

# transform to  ONNX format
success = model.export(format='onnx', task = 'obb', opset = 12)  

                                        