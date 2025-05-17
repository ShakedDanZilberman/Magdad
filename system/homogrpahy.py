import cv2
import numpy as np
import constants
from mouseCamera import MouseCameraHandler as mouse_camera
from ultralytics import YOLO

# model = YOLO('best_new_training.pt')
# model.export(format='openvino', dynamic=True)
# ov_model = YOLO('best_new_training_openvino_model/')
# # results = ov_model('path/to/image.jpg')        # high-level Results object works again


# homogrpahy_matrix, status = cv2.findHomography(constants.SRC_ARRAY_2, constants.DEST_ARRAY_2, 0, 3)

# print(homogrpahy_matrix)


#yolo detect train imgsz=640 data=’data.yaml’ epochs=25 device=’cpu’
#yolo detect train imgsz=320 data="C:/Users/TLP-001/Documents/magdad_project/Magdad/new_yolo/datasets/data.yaml" epochs=25 device=cpu

#yolo detect val model=C:\Users\TLP-001\runs\detect\train\weights\last.pt data=C:\Users\TLP-001\Documents\magdad_project\Magdad\new_yolo\datasets\data.yaml

#yolo detect predict model=C:\Users\TLP-001\runs\detect\train\weights\last.pt source=C:\Users\TLP-001\Documents\magdad_project\Magdad\system\images\20250512_145148.png save=True imgsz=320 conf=0.25 




# ns initialized
# index is 2
# Model loaded
# WARNING Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.
# Loading c:\Users\TLP-001\Magdad\system\best_new_training_openvino_model for OpenVINO inference...
# Using OpenVINO LATENCY mode for batch=1 inference...

# 0: 320x320 (no detections), 108.9ms
# Speed: 1.7ms preprocess, 108.9ms inference, 0.6ms postprocess per image at shape (1, 3, 320, 320)
# eye camera index:  2
# index is 0
# Model loaded
# WARNING Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.
# Loading c:\Users\TLP-001\Magdad\system\best_new_training_openvino_model for OpenVINO inference...
# Using OpenVINO LATENCY mode for batch=1 inference...

# 0: 320x320 (no detections), 50.6ms
# Speed: 1.2ms preprocess, 50.6ms inference, 0.8ms postprocess per image at shape (1, 3, 320, 320)
# eye camera index:  0
# eyes initialized
# running main
# starting gun thread for gun 0
# Gun (193.5, 101.0) is ready to shoot
# Gun 0 is ready to shoot
# Camera 2 is ready
# Camera 0 is ready
# going to loop
# in add yolo
# in add yolo
# history is {(np.float32(138.19179), np.float32(-46.082165)): (1, 279)}
# target stack: [((np.float32(138.19179), np.float32(-46.082165)), 1)]
# angle to shoot:  20.608072
# Rotating to 20.60807228088379 degrees
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.22087), np.float32(-20.788132)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(110.22087), np.float32(-20.788132)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.22087), np.float32(-20.788132)), 1) to gun 0
# target stack: [((np.float32(110.22087), np.float32(-20.788132)), 1), ((np.float32(110.22087), np.float32(-20.788132)), 1)]
# angle to shoot:  34.36445
# Rotating to 34.36444854736328 degrees
# in add yolo
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(111.21823), np.float32(-19.178465)), 1) to gun 0
# target stack: [((np.float32(110.22087), np.float32(-20.788132)), 1), ((np.float32(111.21823), np.float32(-19.178465)), 1)]
# angle to shoot:  34.36445
# Rotating to 34.36444854736328 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(111.21823), np.float32(-19.178465)), 1), ((np.float32(153.77756), np.float32(-44.95241)), 1)]
# angle to shoot:  34.398006
# Rotating to 34.398006439208984 degrees
# in gun rotate: sleep 0.5 secs
# in add yolo
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(153.77756), np.float32(-44.95241)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(153.77756), np.float32(-44.95241)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(153.77756), np.float32(-44.95241)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(153.77756), np.float32(-44.95241)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.98042), np.float32(-44.92429)), 1) to gun 0
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(153.98042), np.float32(-44.92429)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.82115), np.float32(-19.97024)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(153.98042), np.float32(-44.92429)), 1), ((np.float32(110.82115), np.float32(-19.97024)), 1)]
# angle to shoot:  15.153487
# Rotating to 15.153487205505371 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.013824), np.float32(-21.580048)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(110.82115), np.float32(-19.97024)), 1), ((np.float32(110.013824), np.float32(-21.580048)), 1)]
# angle to shoot:  34.35121
# Rotating to 34.35121154785156 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.021904), np.float32(-20.805557)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(110.013824), np.float32(-21.580048)), 1), ((np.float32(110.021904), np.float32(-20.805557)), 1)]
# angle to shoot:  34.257786
# Rotating to 34.25778579711914 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.61841), np.float32(-20.753317)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(110.021904), np.float32(-20.805557)), 1), ((np.float32(110.61841), np.float32(-20.753317)), 1)]
# angle to shoot:  34.424362
# Rotating to 34.42436218261719 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.22087), np.float32(-20.788132)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(110.61841), np.float32(-20.753317)), 1), ((np.float32(110.22087), np.float32(-20.788132)), 1)]
# angle to shoot:  34.244442
# Rotating to 34.244441986083984 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.82115), np.float32(-19.97024)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(110.22087), np.float32(-20.788132)), 1), ((np.float32(110.82115), np.float32(-19.97024)), 1)]
# angle to shoot:  34.36445
# Rotating to 34.36444854736328 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(110.82115), np.float32(-19.97024)), 1) to gun 0
# target stack: [((np.float32(110.82115), np.float32(-19.97024)), 1), ((np.float32(110.82115), np.float32(-19.97024)), 1)]
# angle to shoot:  34.35121
# Rotating to 34.35121154785156 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(138.12206), np.float32(-47.12262)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(110.82115), np.float32(-19.97024)), 1), ((np.float32(138.12206), np.float32(-47.12262)), 1)]
# angle to shoot:  34.35121
# Rotating to 34.35121154785156 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.98042), np.float32(-44.92429)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(138.12206), np.float32(-47.12262)), 1), ((np.float32(153.98042), np.float32(-44.92429)), 1)]
# angle to shoot:  20.499037
# Rotating to 20.49903678894043 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.77756), np.float32(-44.95241)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(153.98042), np.float32(-44.92429)), 1), ((np.float32(153.77756), np.float32(-44.95241)), 1)]
# angle to shoot:  15.153487
# Rotating to 15.153487205505371 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(153.95856), np.float32(-43.92536)), 1) to gun 0
# in add yolo
# in add yolo
# target stack: [((np.float32(153.77756), np.float32(-44.95241)), 1), ((np.float32(153.95856), np.float32(-43.92536)), 1)]
# angle to shoot:  15.224874
# Rotating to 15.224873542785645 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(154.58812), np.float32(-44.840046)), 1) to gun 0
# in add yolo
# target stack: [((np.float32(153.95856), np.float32(-43.92536)), 1), ((np.float32(154.58812), np.float32(-44.840046)), 1)]
# angle to shoot:  15.261131
# Rotating to 15.261131286621094 degrees
# in add yolo
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(154.58812), np.float32(-44.840046)), 1) to gun 0
# in add yolo
# Thread MainThread is alive: True
# Thread ThreadPoolExecutor-0_0 is alive: True
# Thread Thread-2 is alive: True
# Thread Thread-3 is alive: True
# Thread Thread-4 is alive: True
# Exiting...
# target stack: [((np.float32(154.58812), np.float32(-44.840046)), 1), ((np.float32(154.58812), np.float32(-44.840046)), 1)]
# angle to shoot:  14.939195
# Rotating to 14.939194679260254 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# in add yolo
