import cv2
import numpy as np
import constants
from mouseCamera import MouseCameraHandler as mouse_camera

homogrpahy_matrix, status = cv2.findHomography(constants.SRC_ARRAY_2, constants.DEST_ARRAY_2, 0, 3)

print(homogrpahy_matrix)


#yolo detect train imgsz=640 data=’data.yaml’ epochs=25 device=’cpu’
#yolo detect train imgsz=320 data="C:/Users/TLP-001/Documents/magdad_project/Magdad/new_yolo/datasets/data.yaml" epochs=25 device=cpu

#yolo detect val model=C:\Users\TLP-001\runs\detect\train\weights\last.pt data=C:\Users\TLP-001\Documents\magdad_project\Magdad\new_yolo\datasets\data.yaml

#yolo detect predict model=C:\Users\TLP-001\runs\detect\train\weights\last.pt source=C:\Users\TLP-001\Documents\magdad_project\Magdad\system\images\20250512_145148.png save=True imgsz=320 conf=0.25 