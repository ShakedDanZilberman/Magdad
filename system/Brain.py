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
            new_gun = Gun(gun_location, True)
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
        to_init = self.timestep == 5
        if self.timestep % 15 == 14:
            to_check = True
        else:
            to_check = False
        
        eye1 = self.eyes[0]
        targets = eye1.add(to_check, to_init)
        self.add_to_target_list(targets, eye1, MIN_DISTANCE)
        if to_check or to_init:
            print("targets in brain:", self.targets)
        # for eye in self.eyes:
        #     targets = eye.add(to_check, to_init)
        #     self.add_to_target_list(targets, eye, MIN_DISTANCE)
        # # if to_check or to_init:
        #         print("targets in brain:", self.targets)


        
    def too_close(self, target, distance):
        if all(np.linalg.norm(np.array(target) - np.array(old_target)) > distance for old_target in self.targets):
                return False
        return True

    def add_to_target_list(self, new_targets: list, camera: Eye, distance: float = MIN_DISTANCE):
        if new_targets is None:
            return
        for target in new_targets:
            # real_coords = self.calculate_real_coords(target, camera)
            if not self.too_close(target, distance):
                self.add_smart(target, camera)

    def add_smart(self, target, cam):
        # this is temporary:
        self.targets.append(target)
    
    def calculate_real_coords(self, target, camera):
        # for now, we are assuming that the homography matrix is built such that the real coordinates are returned originally.
        return target
        
    def game_loop(self):
        def run_gun():
            gun = self.guns[0]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                print("gun thread running")
                if self.targets is not None:
                    print("Targets in gun thread:", self.targets)   
                    # Get the target coordinates from the last camera
                    target = self.targets[0] # Get the first target from the list
                    print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    slope = (target[0] - gun.gun_location[0]) / (target[1] - gun.gun_location[1])
                    if target[0] - gun.gun_location[0] < 0:
                        angle = np.arctan(slope) * 180 / np.pi + 180
                    else:
                        angle = np.arctan(slope) * 180 / np.pi
                    gun.rotate(angle)
                    gun.shoot()
                    time.sleep(0.5) 


        gun1 = threading.Thread(target=run_gun)
        gun1.start()
        print("Gun 1 is ready to shoot")



        while True:
            self.timestep += 1
            self.add()
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
            # print("targets list size: ", len(self.targets))
            
        cv2.destroyAllWindows()






if __name__=="__main__":
    gun_locations = [(30,48)] # add gun locations here
    cam_info = [(1, (0,0))] # add tuples of (camera index, camera location)
    brain = Brain(gun_locations, cam_info)
    brain.game_loop()
