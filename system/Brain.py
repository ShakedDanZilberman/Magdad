from Eye import Eye
from gun import Gun, DummyGun
from cameraIO import Camera
import threading
from constants import *
import time


class Brain():
    def __init__(self):
        gun1 = Gun()
        gun2 = Gun()
        eye1 = Eye()
        self.guns = [gun1, gun2]
        self.eyes = [eye1]
        self.targets = []
        self.timestep = 0
        coordinates = 

    
    
    def aim_using_targets(self):
        def gun_thread():
            import fit
            print("Gun thread started.")
            global gun_targets
            center = (IMG_WIDTH//2, IMG_HEIGHT//2)
            while True:
                # TODO: test out pop_closest_to_current_location as an alternative to pop() 
                center = target_manager.pop_closest_to_current_location(center)
                # Move the laser pointer to the target
                if center is not None:
                    gun.aim_and_fire_target_2(center)
                    print("Shooting (theoretically)", center)
                    time.sleep(1) # this delay is here so we can wait for the objects to fall and then reset the changes image
                    target_manager.clear() # TODO: reduce the number of frames needed for initialization
        
        gun = threading.Thread(target=gun_thread)
        gun.start()
    
        
    def get_targets(self):
        pass 
    
    def get_eyes(self):
        pass
    
    def get_guns(self):
        pass
    
    def add(self):
        for eye in self.eyes:
            target = eye.add()
        
        
    def too_close(self, target, distance):
        if all(np.linalg.norm(np.array(target) - np.array(old_target)) > distance for old_target in self.targets):
                return True
        return False

    def add_to_target_list(self, new_targets: list, gun: Gun, distance: float = MIN_DISTANCE):
        for target in new_targets:
            if not self.too_close(target, MIN_DISTANCE):
