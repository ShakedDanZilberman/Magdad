from constants import *
from system.cameraIO import Camera
from system.image_processing import RawHandler
from object_finder import Targets


class Eye():
    def __init__(self, camera_location, camera_index):
        self.camera_location = camera_location
        self.camera = Camera(camera_index) 
        self.camera_index = camera_index
        self.raw_handler = RawHandler()
        self.target_manager = Targets()
        self.homography = None
        self.camera_location = 0

    def add(self, to_check, to_init):
        """Add the image to Raw_Handler and Targets.
        This function is called by the main loop to add the image to the handler.
        after adding to the target_manager it calculates the real world coordinates using homography matrix

        Returns:
            real world coordinates of the target
        """
        frame = self.camera.read()
        self.rawHandler.add(frame)
        self.rawHandler.display()
        self.target_manager.add(frame, to_check, to_init)
        if to_check or to_init:
            return self.target_manager.new_targets
        return
    
