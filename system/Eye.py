from constants import *
from system.cameraIO import Camera
from system.image_processing import RawHandler
from object_finder import Targets


class Eye():
    def __init__(self, camera_index):
        self.camera = Camera(camera_index) 
        self.camera_index = camera_index
        self.raw_handler = RawHandler()
        self.target_manager = Targets()
        self.homography = None


    def add(self):
        """Add the image to Raw_Handler and Targets.
        This function is called by the main loop to add the image to the handler.
        after adding to the target_manager it calculates the real world coordinates using homography matrix

        Returns:
            real world coordinates of the target
        """
        pass
        