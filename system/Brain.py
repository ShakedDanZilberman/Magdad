from eyefile import Eye
from gun import Gun, DummyGun
from cameraIO import Camera
import threading
from constants import *
import time
import cv2
import numpy as np

class Brain():
    def __init__(self, gun_locations, cam_info):
        """
        cam info is an array of tuples, for each camera: (cam_index, cam_location)
        gun_locations is an array of tuples, for each gun: its location
        """
        self.guns = []
        for gun_location in gun_locations:
            new_gun = Gun(gun_location)
            self.guns.append(new_gun)
        self.eyes = []
        for cam in cam_info:
            new_eye = Eye(cam[0], cam[1])
            self.eyes.append(new_eye)
        self.targets = []
        self.timestep = 0
    
        
    def get_targets(self):
        pass 
    
    def get_eyes(self):
        pass
    
    def get_guns(self):
        pass
    
    def add(self):
        if self.timestep > 5:
            to_init = False
        else:
            to_init = True
        if self.timestep % 15 == 0:
            to_check = True
        else:
            to_check = False
        
        for eye in self.eyes:
            targets = eye.add(to_check, to_init)
            self.add_to_target_list(targets, eye, MIN_DISTANCE)
        if to_check or to_init:
                print(self.targets)

    def game_loop(self):
        while True:
            self.timestep += 1
            self.add()


        
        
    def too_close(self, target, distance):
        if all(np.linalg.norm(np.array(target) - np.array(old_target)) > distance for old_target in self.targets):
                return True
        return False

    def add_to_target_list(self, new_targets: list, camera: Eye, distance: float = MIN_DISTANCE):
        if new_targets is None:
            return
        for target in new_targets:
            real_coords = self.calculate_real_coords(target, camera)
            if not self.too_close(real_coords, MIN_DISTANCE):
                self.add_smart(real_coords, camera)

    def add_smart(self, target, cam):
        # this is temporary:
        self.targets.append(target)
    
    def calculate_real_coords(self, target, camera):
        # for now, we are assuming that the homography matrix is built such that the real coordinates are returned originally.
        return target
        
    def game_loop(self):
        while True:
            self.timestep += 1
            self.add()
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
            
        cv2.destroyAllWindows()



if __name__=="__main__":
    gun_locations = [] # add gun locations here
    cam_info = [(1, (0,0))] # add tuples of (camera index, camera location)
    brain = Brain(gun_locations, cam_info)
    brain.game_loop()
