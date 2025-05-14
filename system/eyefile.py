from constants import *
from cameraIO import Camera
from image_processing import RawHandler
from object_finder import Targets
import cv2
import numpy as np  
import threading
from mouseCamera import MouseCameraHandler
from yolo import YOLOHandler

class Eye():
    def __init__(self, camera_index, camera_location, homography_matrix):
        self.camera_location = camera_location
        self.camera_index = camera_index
        self.camera = Camera(self.camera_index) 
        self.raw_handler = RawHandler()
        self.target_manager = Targets()
        # self.yolo_handler = YOLOHandler()
        self.homography = homography_matrix
        self.camera_location = 0
        self.real_coords_targets = []
        self.mouse_camera_handler = MouseCameraHandler()
        # uncomment the following lines to add mouse callback to the camera view
        cv2.namedWindow("camera " + str(self.camera_index) + " View")
        cv2.setMouseCallback("camera " + str(self.camera_index) + " View", self.mouse_camera_handler.mouse_callback)

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
        # print("got frame")
        self.raw_handler.add(frame)
        self.mouse_camera_handler.add(frame)
        self.mouse_camera_handler.display(self.camera_index)
        if self.mouse_camera_handler.has_new_clicks():
            click_positions = self.mouse_camera_handler.get_clicks()
            # print(f"camera {self.camera_index}: Clicked position in pixels:", click_positions)
            pixel_coords = np.array(click_positions, dtype='float32').reshape(-1, 1, 2)
            real_coords_array = cv2.perspectiveTransform(pixel_coords, self.homography)
            x = real_coords_array[0][0][0]
            y = real_coords_array[0][0][1]
            return [(x, y)]
            # return real_coords_array
        return []
        # print(f"coords in reality: {self.real_coords_targets}")

            # print(f"added real coords: {self.real_coords_targets}")

    def add_yolo(self, frame, to_check=True):
        """
        Add the image to Raw_Handler and Targets.
        This function is called by the main loop to add the image to the handler.
        after adding to the target_manager it calculates the real world coordinates using homography matrix

        Returns:
            real world coordinates of the target
        """
        # frame = self.camera.read()
        # self.raw_handler.add(frame)
        # # self.raw_handler.display(self.camera_index)
        self.yolo_handler.add(frame)
        # self.yolo_handler.display()
        if to_check:
            pixel_coords = np.array(self.yolo_handler.get_centers(), dtype='float32').reshape(-1, 1, 2)
            real_coords_array = cv2.perspectiveTransform(pixel_coords, self.homography)
            if real_coords_array is not None:
                real_coords_targets = [tuple(pt[0]) for pt in real_coords_array]
                return real_coords_targets
        return []
    
    def display(self):
        """
        Display the image stored in the handler.
        Uses cv2.imshow() to display the image.
        """
        # self.raw_handler.display(self.camera_index)
        # self.mouse_camera_handler.display(self.camera_index)
        self.yolo_handler.display()

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




    
