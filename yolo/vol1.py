from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolov8.pt", "yolov11n.pt", device="cpu")
# Perform object detection on an image
results = model("yolo\images\Screenshot 2025-01-09 204222.jpg")
# Visualize the results
results[0].show()
# Export the model to ONNX format
#success = model.export(format="onnx")