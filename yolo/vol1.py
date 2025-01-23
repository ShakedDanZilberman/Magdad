from pathlib import Path
import yaml
import os
from io import BytesIO
from ultralytics import YOLO
import torch
dependencies = ['torch']
from torchvision.models.resnet import resnet18 as _resnet18


model = YOLO('C:/Users/TLP-001/runs/detect/train26/weights/last.pt')

# Perform prediction
results = model.predict(
    source='C:/Users/TLP-001/Documents/magdad_project/Magdad/yolo/images/img4.jpg',
    save=True,
    imgsz=640,
    conf=0.25
)















# from roboflow import Roboflow
# from ultralytics import YOLO


# # rf = Roboflow(api_key='tYdBc3fC65ADwcIsYT2D')
# # project = rf.workspace().project('camera-training-ltw8l')
# # # Download the dataset
# # model = project.version(2).download('yolov8')

# # Train the model
# model = YOLO('yolov8s.yaml')
# model.train(data='dataset/data.yaml', epochs=25, imgsz=640)



# Path to your dataset

#dataset_path = r"C:\Users\TLP-001\Documents\magdad_project\Magdad\yolo\dataset\data.yaml"
# # Check if the dataset file exists
# if not dataset_path.exists():
#     raise FileNotFoundError(f"Dataset file {dataset_path} does not exist")

# Load the dataset configuration
# with open(dataset_path, 'rb') as file:
#     print(file.read())
#     dataset_config = yaml.safe_load(file)

# print("Current working directory:", os.getcwd())


# Assuming you are using a YOLO-compatible library

# Load the model configuration and weights (if you have pre-trained weights)
# model = YOLO('yolov5s.yaml')  # Adjust the model config file as needed

# # Train the model
# model.train(
#     data='data.yaml',
#     epochs=25,
#     imgsz=640,  # Might need to reduce if memory issues occur
#     device='cpu'
# )



#train


#validate


# Predict on a single image
# yolo detect predict model=C:\Users\TLP-001\runs\detect\train26\weights\last.pt data=C:\Users\TLP-001\Documents\magdad_project\Magdad\yolo\dataset\data.yaml source=C:\Users\TLP-001\Documents\magdad_project\Magdad\yolo\images\img6.jpg save=True imgsz=640 conf=0.25

# yolo detect predict model=C:\Users\TLP-001\runs\detect\train26\weights\last.pt source=C:\Users\TLP-001\Documents\magdad_project\Magdad\yolo\images\img6.jpg save=True imgsz=640 conf=0.25


