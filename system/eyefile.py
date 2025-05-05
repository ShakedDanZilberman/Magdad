from constants import *
from cameraIO import Camera
from image_processing import RawHandler
from object_finder import Targets
import cv2
import numpy as np  
import threading

class Eye():
    def __init__(self, camera_index, camera_location, homography_matrix):
        self.camera_location = camera_location
        self.camera_index = camera_index
        self.camera = Camera(self.camera_index) 
        self.raw_handler = RawHandler()
        self.target_manager = Targets()
        self.homography = homography_matrix
        self.camera_location = 0
        self.real_coords_targets = []

    def add(self, to_check, to_init):
        """Add the image to Raw_Handler and Targets.
        This function is called by the main loop to add the image to the handler.
        after adding to the target_manager it calculates the real world coordinates using homography matrix

        Returns:
            real world coordinates of the target
        """
        frame = self.camera.read()
        self.raw_handler.add(frame)
        self.raw_handler.display(self.camera_index)
        
        self.target_manager.add(frame, to_check=to_check, to_init=to_init)
        if to_check or to_init:
            pixel_coords = np.array(self.target_manager.new_targets, dtype='float32').reshape(-1, 1, 2)
            self.real_coords_targets = cv2.perspectiveTransform(pixel_coords, self.homography)

            
        return
    
    
# class Eye:
#     def __init__(self, CAMERA_INDEX_0, camera_location):
#         self.camera_location = camera_location
#         self.CAMERA_INDEX_0 = CAMERA_INDEX_0
#         self.camera = Camera(CAMERA_INDEX_0)
#         self.raw_handler = RawHandler()
#         self.target_manager = Targets()
#         self.frame_lock = threading.Lock()
#         self.latest_frame = None

#     def get_frame(self):
#         with self.frame_lock:
#             if self.latest_frame is None:
#                 return None
#             return self.latest_frame.copy()

#     def add(self, frame, to_check, to_init):
#         self.raw_handler.add(frame)
#         self.raw_handler.display()
#         self.target_manager.add(frame, to_check=to_check, to_init=to_init)
#         if to_check or to_init:
#             return self.target_manager.new_targets
#         return []



    
