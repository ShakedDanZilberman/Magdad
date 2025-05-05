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
        cam info is an array of tuples, for each camera: (cam_index, CAMERA_LOCATION_0)
        gun_locations is an array of tuples, for each gun: its location
        """
        self.guns = []
        for gun_location in gun_locations:
            new_gun = Gun(gun_location, True)
            self.guns.append(new_gun)
        self.eyes = []
        for cam in cam_info:
            new_eye = Eye(cam[0], cam[1], cam[2])
            self.eyes.append(new_eye)
        self.targets = []
        self.timestep = 0
        


    def get_targets(self):
        return self.targets
    
    def get_eyes(self):
        return self.eyes
    
    def get_guns(self):
        return self.guns
    
    def add(self):
        to_init = self.timestep == 5
        if self.timestep % 15 == 14:
            to_check = True
        else:
            to_check = False
        
        # eye1 = self.eyes[0]
        # targets = eye1.add(to_check, to_init)
        # self.add_to_target_list(targets, eye1, MIN_DISTANCE)
        # if to_check or to_init:
        #     print("targets in brain:", self.targets)
        for eye in self.eyes:
            targets = eye.add(to_check, to_init)
            self.add_to_target_list(targets, eye, MIN_DISTANCE)
        if to_check or to_init:
            print("targets in brain:", self.targets)

    def too_close(self, target, distance):
        if all(
            np.linalg.norm(np.array(target) - np.array(old_target)) > distance
            for old_target in self.targets
        ):
            return False
        return True

    def add_to_target_list(
        self, new_targets: list, camera: Eye, distance: float = MIN_DISTANCE
    ):
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

    def calculate_angle(self, target, gun_index=0):
        gun = self.guns[gun_index]
        if target[0] < gun.gun_location[0]:
            slope = (gun.gun_location[0]- target[0]) / (gun.gun_location[1] - target[1])
            angle = np.arctan(slope) * 180 / np.pi
            angle = angle * (-1)
        else:
            slope = (gun.gun_location[1]- target[1]) / (target[0] - gun.gun_location[0])
            angle = 90 - np.arctan(slope) * 180 / np.pi
        return angle

    def game_loop(self):
        print("running main")
        def run_gun():
            gun = self.guns[0]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                if self.targets is not None and len(self.targets)>0:
                    # Get the target coordinates from the last camera
                    target = self.targets.pop(0)              
                    print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    angle = self.calculate_angle(target)
                    print("angle to shoot: ", angle)
                    gun.rotate(angle*(-1))
                    gun.shoot()
                    time.sleep(0.5)

        # gun1 = threading.Thread(target=run_gun)
        # gun1.start()
        print("Gun 1 is ready to shoot")
        
        while True:
            self.timestep += 1
            self.add()
            # print("timestep: ", self.timestep)
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def game_loop_independent(self):
        """
        This function is the main loop of the program. It runs independently of image procesing.
        the user spots the targets and the gun shoots at them.
        """
        print("running main")
        def run_gun():
            gun = self.guns[0]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                if self.targets is not None and len(self.targets)>0:
                    # Get the target coordinates from the last camera
                    target = self.targets.pop(0)              
                    print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    angle = self.calculate_angle(target)
                    print("angle to shoot: ", angle)
                    gun.rotate(angle*(-1))
                    gun.shoot()
                    time.sleep(0.5)

        # gun1 = threading.Thread(target=run_gun)
        # gun1.start()
        print("Gun 1 is ready to shoot")
        
        while True:
            self.timestep += 1
            self.add()
            # print("timestep: ", self.timestep)
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gun_locations = []  # example
    # cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    Brain(gun_locations, cam_info).game_loop_independent()


# class Brain:
#     def __init__(self, gun_locations, cam_info):
#         """
#         cam_info: list of (cam_index, CAMERA_LOCATION_0)
#         gun_locations: list of gun positions
#         """
#         self.guns = [Gun(loc, True) for loc in gun_locations]
#         self.eyes = [Eye(cam_idx, cam_loc) for cam_idx, cam_loc in cam_info]
#         self.targets = []
#         self.timestep = 0
#         self.running = True

#     def add(self):
#         to_init = (self.timestep == 5)
#         to_check = (self.timestep % 15 == 14)
#         for eye in self.eyes:
#             # get latest frame copy
#             frame = eye.get_frame()
#             if frame is None:
#                 continue
#             new = eye.add(frame, to_check, to_init)
#             self._merge_targets(new)
#         if to_check or to_init:
#             print("targets in brain:", self.targets)

#     def _merge_targets(self, new_targets):
#         if not new_targets:
#             return
#         for t in new_targets:
#             if not self._too_close(t, MIN_DISTANCE):
#                 self.targets.append(t)

#     def _too_close(self, tgt, dist):
#         return any(np.linalg.norm(np.array(tgt)-np.array(old)) <= dist for old in self.targets)

#     def calculate_angle(self, target, gun_index=0):
#         gun = self.guns[gun_index]
#         dx = target[0] - gun.gun_location[0]
#         dy = target[1] - gun.gun_location[1]
#         angle = np.degrees(np.arctan2(dy, dx))
#         return angle

#     def game_loop(self):
#         print("running main")
#         # start producer threads for each camera
#         for idx, eye in enumerate(self.eyes):
#             t = threading.Thread(target=self._frame_producer, args=(idx,), daemon=True)
#             t.start()
#         # display threads
#         for idx, eye in enumerate(self.eyes):
#             t = threading.Thread(target=self._show_display, args=(idx,), daemon=True)
#             t.start()
#         # gun thread(s)
#         # gun_t = threading.Thread(target=self._run_gun, daemon=True)
#         # gun_t.start()

#         # main loop: target processing
#         try:
#             while self.running:
#                 self.timestep += 1
#                 self.add()
#                 time.sleep(0.02)
#         except KeyboardInterrupt:
#             self.running = False
#             cv2.destroyAllWindows()

#     def _frame_producer(self, cam_idx):
#         print(f"frame producer {cam_idx} started")
#         eye = self.eyes[cam_idx]
#         while self.running:
#             frame = eye.camera.read()
#             with eye.frame_lock:
#                 eye.latest_frame = frame.copy()
#             time.sleep(0.001)

#     def _show_display(self, cam_idx):
#         print(f"display {cam_idx} started")
#         eye = self.eyes[cam_idx]
#         while self.running:
#             with eye.frame_lock:
#                 if eye.latest_frame is None:
#                     continue
#                 frame = eye.latest_frame.copy()
#             eye.raw_handler.add(frame)
#             eye.raw_handler.display()
#             if cv2.waitKey(1) == 27:
#                 self.running = False
#                 break

#     def _run_gun(self):
#         gun = self.guns[0]
#         print(f"Gun {gun.gun_location} is ready to shoot")
#         while self.running:
#             if self.targets:
#                 tgt = self.targets.pop(0)
#                 angle = self.calculate_angle(tgt)
#                 gun.rotate(-angle)
#                 gun.shoot()
#                 time.sleep(0.5)





