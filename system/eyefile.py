from constants import *
from cameraIO import Camera
from image_processing import RawHandler
from object_finder import Targets
import cv2
import numpy as np  
import threading
from mouseCamera import MouseCameraHandler

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
        self.mouse_camera_handler = MouseCameraHandler()
        cv2.namedWindow("camera " + str(self.camera_index) + " view")
        cv2.setMouseCallback("camera " + str(self.camera_index) + " view", self.mouse_camera_handler.mouse_callback)

    def get_real_coords_targets(self):
        """
        Get the real world coordinates of the targets.
        This function is called by the main loop to get the real world coordinates of the targets.
        Returns:
            real world coordinates of the targets
        """
        return self.real_coords_targets

    def add(self, to_check, to_init):
        """
        Add the image to Raw_Handler and Targets.
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
            real_coords_array = cv2.perspectiveTransform(pixel_coords, self.homography)
            self.real_coords_targets = [tuple(pt[0]) for pt in real_coords_array]
            
#         return
    

    def add_independent(self):
        """
        Add the image to Raw_Handler but not to Targets.
        
        """
        frame = self.camera.read()
        self.mouse_camera_handler.add(frame)
        self.mouse_camera_handler.display(self.camera_index)
        if self.mouse_camera_handler.has_new_click():
            click_pos = self.mouse_camera_handler.get_last_click()
            print(f"camera {self.camera_index}: Clicked position in pixels:", click_pos)
            pixel_coords = np.array([click_pos], dtype='float32').reshape(-1, 1, 2)
            print(f"coords in pixels {pixel_coords}, type: {type(pixel_coords)}, homography {self.homography}, type: {type(self.homography)}")
            real_coords_array = cv2.perspectiveTransform(pixel_coords, self.homography)
            self.real_coords_targets = [tuple(pt[0]) for pt in real_coords_array]
            print(f"added real coords: {self.real_coords_targets}")

    
# import threading

# class Eye():
#     def __init__(self, camera_index, camera_location):
#         self.camera_location = camera_location
#         self.camera_index = camera_index
#         self.camera = Camera(self.camera_index) 
#         self.raw_handler = RawHandler()
#         self.target_manager = Targets()
#         self.homography = None

#         self.latest_frame = None
#         self.frame_lock = threading.Lock()
#         self.running = True

#         # Start frame update thread
#         self.capture_thread = threading.Thread(target=self._frame_producer, daemon=True)
#         self.capture_thread.start()

#     def _frame_producer(self):
#         while self.running:
#             frame = self.camera.read()
#             with self.frame_lock:
#                 self.latest_frame = frame.copy()

#     def get_frame(self):
#         with self.frame_lock:
#             if self.latest_frame is None:
#                 return None
#             return self.latest_frame.copy()

#     def add(self, frame, to_check, to_init):
#         self.target_manager.add(frame, to_check=to_check, to_init=to_init)
#         if to_check or to_init:
#             return self.target_manager.new_targets
#         return




    
