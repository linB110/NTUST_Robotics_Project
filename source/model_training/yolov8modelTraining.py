# %% 
# Import necessary libraries
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Necessary for multiprocessing support (Windows environment)

    # Load YOLOv8 OBB model
    model = YOLO('D:/yoloV8/yolov8x-obb.pt')  # Use the OBB model, dataset must be obb format

    # Start training
    results = model.train(
        data='D:/yoloV8/data.yaml',
        imgsz=640,
        epochs=300,  # run times
        patience=30, # early break
        batch=16,
        label_smoothing=0.15,
        lr0 = 0.001, # ini learing rate
        lrf = 0.01, # final learing rate
        weight_decay=5e-4,
        # device=0,  # Specify GPU device index (use CPU if not specified)
        amp=True,  # Enable Automatic Mixed Precision       
        project='yolov8x_obbmodel',  # Project name
        name='obb_model',  # Experiment name
        optimizer='SGD'
    )

    # Optional: Evaluate the model after training
    try:
        print("Evaluating the model...")
        metrics = model.val(data='D:/yoloV8/train_ori/data.yaml')  # Ensure using the correct data config
        print("Evaluation metrics:", metrics)
    except Exception as e:
        print(f"Error during evaluation: {e}")
