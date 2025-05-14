# import cv2
# import numpy as np
# from constants import *
# from mouseCamera import MouseCameraHandler as mouse_camera
# import os

# # homogrpahy_matrix, status = cv2.findHomography(constants.SRC_ARRAY_0, constants.DEST_ARRAY_0, 0, 3)

# # print(homogrpahy_matrix)


# from ultralytics import YOLO
# from openvino.runtime import Core
# from ultralytics.data.augment import LetterBox

# # # Resolve model path
# # base = os.path.dirname(os.path.abspath(__file__))
# # model = YOLO(os.path.join(base, 'best_new_training.pt'))
# # model.export(format='openvino')
# # imgsz = imgsz
# # conf_threshold = 0.5
# # img = None
# # bounding_boxes = []
# # letterbox = LetterBox(new_shape=(imgsz, imgsz))
# # # inside YOLOHandler.__init__:
# # dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
# # _ = self.model(dummy, imgsz=imgsz)


# import cv2
# import numpy as np
# from openvino.runtime import Core

# # Initialize OpenVINO Core
# ie = Core()

# # Load the OpenVINO model (XML and BIN files)
# model_path = 'system/best_new_training_openvino_model/best_new_training.xml'
# compiled_model = ie.compile_model(model=model_path, device_name="CPU")
# output_layer = compiled_model.output(0)
# print(f"Output layer: {output_layer}")
# # Load an image for testing
# image_path = "system/images/20250512_144844.png"  # Replace with your test image path
# image = cv2.imread(image_path)
# input_image = cv2.resize(image, (320, 320))  # Resize to the model input size
# input_image = input_image / 255.0  # Normalize
# input_image = input_image.transpose(2, 0, 1)  # Convert HWC to CHW format
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# # Run inference
# results = compiled_model([input_image])[output_layer]
# print("Results:", results)
# # Decode and visualize bounding boxes
# for detection in results[0]:
#     confidence = detection[4]
#     if confidence > 0.3:  # Set a confidence threshold (adjustable)
#         x_center, y_center, width, height = detection[0:4]
#         x_center *= IMG_WIDTH
#         y_center *= IMG_HEIGHT
#         width *= IMG_WIDTH
#         height *= IMG_HEIGHT
        
#         # Calculate the top-left corner
#         x1 = int(x_center - width / 2)
#         y1 = int(y_center - height / 2)
#         x2 = int(x_center + width / 2)
#         y2 = int(y_center + height / 2)
        
#         # Draw the bounding box on the image
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Display the image with bounding boxes
# cv2.imshow("YOLOv8 OpenVINO Inference", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
