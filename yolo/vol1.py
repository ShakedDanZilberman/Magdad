from roboflow import Roboflow


rf = Roboflow(api_key='tYdBc3fC65ADwcIsYT2D')
project = rf.workspace('ayalulu').project('camera-training-ltw8l')
# Download the dataset
dataset = project.version(1).download('yolov8')
print("finished downloading dataset")
# from ultralytics import YOLO

# #Load a pretrained YOLO model
# model = YOLO("yolo\yolov8m-seg.pt")
# # Perform object detection on an image
# results = model("yolo\images\Screenshot 2025-01-09 204149.jpg")
# # Visualize the results
# results[0].show()

