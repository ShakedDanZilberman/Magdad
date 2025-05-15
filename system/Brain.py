from eyefile import Eye
from gun import Gun, DummyGun
from cameraIO import Camera
import threading
from constants import *
import time
import numpy as np
import queue
# from vis_production import CameraProducer


from import_defence import ImportDefence

with ImportDefence():
    import openvino
    import cv2


class CameraProducer(threading.Thread):
    def __init__(self, camera_index, eye, output_queue):
        super().__init__(daemon=True)
        # self.cap = cv2.VideoCapture(camera_index)
        self.eye = eye
        self.output_queue = output_queue
        self.win_name = f"Cam {camera_index}"
        self.timestep = 0

    def run(self):
        while True:
            self.timestep+=1
            # print(f"timestep: {self.timestep} camera {self.eye.camera_index}")
            frame = self.eye.camera.read(self.timestep)
            # Run your add_yolo processing (which returns targets)
            if self.timestep % 30 == 14:
                print("in add yolo")
                targets = self.eye.add_yolo(frame)
            else:
                targets = self.eye.add_yolo(frame, False)
            # Grab the visualized image (after display())
            self.eye.yolo_handler.prepare_to_show()
            vis = self.eye.yolo_handler.get_vis()
            # print(f"visual is {vis}")
            # Put into queue: window name, image, AND targets
            item = (self.win_name, vis, targets)
            try:
                self.output_queue.put(item, block=False)
            except queue.Full:
                # drop oldest if full
                _ = self.output_queue.get_nowait()
                self.output_queue.put(item, block=False)

            time.sleep(0.01)

class DisplayConsumer(threading.Thread):
    def __init__(self, input_queue):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.latest = {}  # win_name -> frame

    def run(self):
        while True:
            # Harvest all available frames
            while True:
                try:
                    win_name, frame = self.input_queue.get(block=False)
                    self.latest[win_name] = frame
                except queue.Empty:
                    break

            # Display all latest
            for win_name, frame in self.latest.items():
                cv2.imshow(win_name, frame)

            # One waitKey to refresh all windows
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                return

class Brain():
    def __init__(self, gun_info, cam_info):
        """
        cam info is an array of tuples, for each camera: (cam_index, CAMERA_LOCATION_0)
        gun_locations is an array of tuples, for each gun: its location
        """
        self.guns = []
        for gun in gun_info:
            new_gun = Gun(gun[0], gun[1], False)
            self.guns.append(new_gun)
        print("guns initialized")
        self.eyes = []
        for cam in cam_info:
            new_eye = Eye(cam[0], cam[1], cam[2])
            print("eye camera index: ", cam[0])
            self.eyes.append(new_eye)
        print("eyes initialized")
        self.targets = {}
        self.history = {}
        self.timestep = 0
        self.display_queue = queue.Queue(maxsize=len(self.eyes))
        self.latest = {}

    def get_targets(self):
        return self.targets
    
    def get_eyes(self):
        return self.eyes
    
    def get_guns(self):
        return self.guns
    
    def too_close(self, target, min_dist):
        """
        Check if `target` (x, y) is within `min_dist` of any existing target.
        Returns True if too close to an existing target.
        """
        tx, ty = target
        # print("tx, ty:", (tx, ty))
        for (ox, oy) in self.targets.keys():
            # print("ox, oy:", (ox, oy))
            if np.linalg.norm((tx - ox, ty - oy)) <= min_dist:
                return True
        for (ox, oy) in self.history.keys():
            if np.linalg.norm((tx - ox, ty - oy)) <= min_dist:
                return True
        return False

    def add_to_target_list(self, new_targets, distance: float = MIN_DISTANCE):
        # print("new targets: ", new_targets)
        if len(new_targets) == 0:
            return
        for targ in new_targets:
            # print("targ: ", targ, "distance: ", distance)
            if not self.too_close(targ, distance):
                self.add_smart(targ)

    def find_priority(self, target_location):
        # for now the priority is set to be a factor of the distance to the target
        x, y = target_location
        # Scale y (0-50) to priority (1-10)
        prio = int(np.clip((y / 50) * 10, 1, 10)) 
        return prio

    def add_smart(self, target_location):
        """
        Compute priority based on y-value of target and add to dict.
        Priority ranges 1 (lowest) to 10 (highest); y goes 0..50.
        """
        priority = self.find_priority(target_location)
        # print("adding target: ", target_location, "priority: ", priority)
        self.targets[target_location] = priority
        # print("targets: ", self.targets)
        
    def add(self):
        to_init = self.timestep == 5
        if self.timestep % 15 == 14:
            to_check = True
        else:
            to_check = False
        for eye in self.eyes:
            eye.add(to_check, to_init)
            targets = eye.get_real_coords_targets()
            self.add_to_target_list(targets, MIN_DISTANCE)
        if to_check or to_init:
            print("targets in brain:", self.targets)

    # def too_close(self, target, distance):
    #     if all(
    #         np.linalg.norm(np.array(target) - np.array(old_target)) > distance
    #         for old_target in self.targets
    #     ):
    #         return False
    #     return True

    # def add_to_target_list(
    #     self, new_targets: list, camera: Eye, distance: float = MIN_DISTANCE
    # ):
    #     if new_targets is None:
    #         return
    #     for target in new_targets:
    #         # real_coords = self.calculate_real_coords(target, camera)
    #         if not self.too_close(target, distance):
    #             self.add_smart(target, camera)

    # def add_smart(self, target, cam):
    #     # this is temporary:
    #     self.targets.append(target)

    def calculate_angle_from_gun(self, target, gun_index=0):
        gun = self.guns[gun_index]
        # print(f"gun location: {gun.gun_location}, target: {target}")
        if target[0] < gun.gun_location[0]:
            slope = (gun.gun_location[0]- target[0]) / (gun.gun_location[1] - target[1])
            angle = np.arctan(slope) * 180 / np.pi
            angle = angle * (-1)
        else:
            # print("in calculate angle: gun location: ", gun.gun_location, "target: ", target)
            slope = (gun.gun_location[1]- target[1]) / (target[0] - gun.gun_location[0])
            angle = 90 - np.arctan(slope) * 180 / np.pi
        return angle*(-1)

    def calculate_angle(self, location_1: tuple, location_2: tuple):
        """
        This function calculates the angle between two points in degrees.
        It returns the angle in degrees.
        """
        x1, y1 = location_1
        x2, y2 = location_2
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        return np.abs(angle)
        
    def game_loop(self):
        print("running main")
        def run_gun(index):
            gun = self.guns[index]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                if self.targets is not None and len(self.targets)>0:
                    # Get the target coordinates from the last camera
                    target = self.targets.pop(0)              
                    print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    angle = self.calculate_angle_from_gun(target[0], index)
                    gun.rotate(angle)
                    gun.shoot()
                    time.sleep(0.5)

        for i in range(len(self.guns)):
            # Start a thread for each gun   
            gun1 = threading.Thread(target=run_gun, args=(i,))
            gun1.start()
            print(f"Gun {i} is ready to shoot")
        
        while True:
            self.timestep += 1
            self.add()
            # print("timestep: ", self.timestep)
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def add_independent(self):
        """
        this function deals with frame adding without using yolo
        it allows the user to add targets to the list of targets by clicking them on the screen.
        after clicking, the target is added to the list of targets and the gun shoots at it.
        """
        
        for eye in self.eyes:
            targets  = eye.add_independent()
            if len(targets) > 0:
                self.add_to_target_list(targets, MIN_DISTANCE)

    def add_yolo(self):
        """
        this function deals with frame adding without using yolo
        it allows the user to add targets to the list of targets by clicking them on the screen.
        after clicking, the target is added to the list of targets and the gun shoots at it.
        """
        # to_init = self.timestep == 5
        # if self.timestep % 15 == 14:
        #     to_check = True
        # else:
        #     to_check = False
        for eye in self.eyes:
            targets = eye.add_yolo(True, True)
            if len(targets) > 0:
                self.add_to_target_list(targets, MIN_DISTANCE)

    def check_emergency_targets(self):
        emergency_targets = []
        for location, priority in self.targets.items():
            if priority >= 9:
                emergency_targets.append((location, priority))
        for i, gun in enumerate(self.guns):
            if len(emergency_targets) > 0:
                target = emergency_targets[i]
                # if there is an emergency target, assign it to the first gun
                if len(gun.target_stack) > 1:
                    if gun.target_stack[1][1] < target[1]:
                        old_next = gun.target_stack.pop(1)
                        self.add_to_target_list([old_next])
                gun.target_stack.append(target)
                

    def pop_optimized(self, gun):
        min_anglular_distance = float("inf")
        closest_target = None
        if len(self.targets) == 0:
            return 
        elif len(gun.target_stack) == 0:
            target = self.targets.popitem()
            gun.target_stack.append(target)
            timestep = self.timestep
            self.history[target[0]] = (target[1], timestep)
            print(f"history is {self.history}")
            return
        for location, priority in self.targets.items():
            if gun.target_stack is not None and len(gun.target_stack) > 0:
                angular_dist = self.angle_diff(gun.target_stack[0][0], location, gun.gun_index)
                if angular_dist < min_anglular_distance:
                    min_anglular_distance = angular_dist
                    closest_target = (location, priority)
                    # print(f"pop_optimized: closest target: {closest_target}, angular distance: {angular_dist}")
                    # print(f"pop_optimized: gun location: {gun.gun_location}, target: {location}, angular distance: {angular_dist}")
        if closest_target is not None:
            # print("closest target: ", closest_target)
            priority = self.targets.pop(closest_target[0])
            # print(f"targets in brain after pop: {self.targets}")
            gun.target_stack.append((closest_target[0], priority))
            print(f"assigned target {closest_target} to gun {gun.gun_index}")

    def angle_diff(self, location_1: tuple, location_2: tuple, gun_index=0):
        """
        This function calculates the difference between two angles.
        It returns the difference in degrees.
        """
        # print("location 1: ", location_1, "location 2: ", location_2)
        angle1 = self.calculate_angle_from_gun(location_1, gun_index)
        angle2 = self.calculate_angle_from_gun(location_2, gun_index)
        diff = angle1 - angle2
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return np.abs(diff)

    def update_history(self):
        """
        This function updates the history of targets.
        It removes targets that are not in the list of targets anymore.
        """
        for location, data in self.history.items():
            if data[1] - self.timestep > HISTORY_DELAY:
                self.history.pop(location)
                # print("history: ", self.history)
                # print("targets: ", self.targets)
                # print("history after pop: ", self.history)

    # def pop_optimized(self, gun_location):
    #     """
    #     This function pops the target from the list of targets that is closest to the gun location.
    #     It returns the target and removes it from the list of targets.
    #     If there are no targets, it returns None.
    #     gun_location is a tuple (x, y) of the gun location.
    #     """
    #     if len(self.targets) == 0:
    #         return None
    #     closest_target = None
    #     closest_distance = float("inf")
    #     for location, priority in self.targets:
    #         distance = np.linalg.norm(np.array(location) - np.array(gun_location))
    #         if distance < closest_distance:
    #             closest_distance = distance
    #             closest_target = location
    #     self.targets.pop(closest_target)
    #     return closest_target

    def game_loop_independent(self):
        """
        This function is the main loop of the program. It runs independently of image procesing.
        the user spots the targets and the gun shoots at them.
        """
        print("running main")
        def run_gun(gun_index):
            gun = self.guns[gun_index]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                if gun.target_stack is not None and len(gun.target_stack)>0:
                    # print("gun target stack: ", gun.target_stack)
                    # Get the target coordinates from the last camera
                    target = gun.target_stack[0]            
                    # print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    print(f"target stack: {gun.target_stack}")
                    angle = self.calculate_angle_from_gun(target[0], gun_index)
                    # print("angle to shoot: ", angle)
                    gun.rotate(angle)
                    gun.shoot()
                    print("shot fired")
                    gun.target_stack.pop(0)
                    # print("gun target stack after pop: ", gun.target_stack)
                    time.sleep(0.1)


            # check which gun is free and assign it to the target
            # if there is a target in the list of targets that has priority 8 or higher, assign it to the first gun immediately
        
        for i, gun in enumerate(self.guns):
            print(f"starting gun thread for gun {i}")
            gun_thread = threading.Thread(target=run_gun, args=(i,))
            gun_thread.start() 
            print(f"Gun {i} is ready to shoot")

        while True:
            self.timestep += 1
            # print("timestep: ", self.timestep)
            self.add_independent()
            self.check_emergency_targets()
            # print(f"targets in brain: {self.targets}")
            for gun in self.guns:
                
                if gun.is_free():
                    # assign the target to the gun
                    self.pop_optimized(gun)
                    
            # print("timestep: ", self.timestep)
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def game_loop_yolo(self):
        """
        This function is the main loop of the program. It runs independently of image procesing.
        the user spots the targets and the gun shoots at them.
        """
        print("running main")
        cv2.startWindowThread()
        def run_gun(gun_index):
            gun = self.guns[gun_index]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                if gun.target_stack is not None and len(gun.target_stack)>0:
                    # print("gun target stack: ", gun.target_stack)
                    # Get the target coordinates from the last camera
                    target = gun.target_stack[0]            
                    # print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    print(f"target stack: {gun.target_stack}")
                    angle = self.calculate_angle_from_gun(target[0], gun_index)
                    # print("angle to shoot: ", angle)
                    gun.rotate(angle)
                    gun.shoot()
                    print("shot fired")
                    gun.target_stack.pop(0)
                    # print("gun target stack after pop: ", gun.target_stack)
                    time.sleep(0.1)
            # check which gun is free and assign it to the target
            # if there is a target in the list of targets that has priority 8 or higher, assign it to the first gun immediately
        
        for i, gun in enumerate(self.guns):
            print(f"starting gun thread for gun {i}")
            gun_thread = threading.Thread(target=run_gun, args=(i,))
            gun_thread.start() 
            print(f"Gun {i} is ready to shoot")


        while True:
            self.timestep += 1
            # print("timestep: ", self.timestep)
            self.add_yolo()
            self.check_emergency_targets()
            # print(f"targets in brain: {self.targets}")
            for eye in self.eyes:
                eye.display()
            
            for gun in self.guns:
                
                if gun.is_free():
                    # assign the target to the gun
                    self.pop_optimized(gun)
                    self.update_history()
                    # print(f"gun {gun.gun_index} target stack: ", gun.target_stack)
            # print("timestep: ", self.timestep)
            # Press Escape to exit
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def game_loop_display(self):
        """
        This function is the main loop of the program. It runs independently of image procesing.
        the user spots the targets and the gun shoots at them.
        """
        print("running main")
        # cv2.startWindowThread()
        def run_gun(gun_index):
            gun = self.guns[gun_index]
            # This function will run in a separate thread for each gun
            print(f"Gun {gun.gun_location} is ready to shoot")
            while True:
                if gun.target_stack is not None and len(gun.target_stack)>0:
                    # print("gun target stack: ", gun.target_stack)
                    # Get the target coordinates from the last camera
                    target = gun.target_stack[0]            
                    # print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    print(f"target stack: {gun.target_stack}")
                    angle = self.calculate_angle_from_gun(target[0], gun_index)
                    print("angle to shoot: ", angle)
                    if angle - gun.current_angle:
                        gun.rotate(angle)
                        gun.shoot()
                        # print("shot fired")
                        gun.target_stack.pop(0)
                    # print("gun target stack after pop: ", gun.target_stack)
                    print("in gun thread: sleeping for 1 second")
                    time.sleep(1)
            # check which gun is free and assign it to the target
            # if there is a target in the list of targets that has priority 8 or higher, assign it to the first gun immediately
        
        for i, gun in enumerate(self.guns):
            print(f"starting gun thread for gun {i}")
            gun_thread = threading.Thread(target=run_gun, args=(i,))
            gun_thread.start() 
            print(f"Gun {i} is ready to shoot")

        # Create a queue for the display thread
        producers = []
        for eye in self.eyes:
            prod = CameraProducer(eye.camera_index, eye, self.display_queue)
            prod.start()
            print(f"Camera {eye.camera_index} is ready")
            producers.append(prod)

        print("going to loop")
        while True:
            self.timestep += 1
            # 1. Add targets from cameras
            while True:
                try:
                    win_name, frame, targets = self.display_queue.get(block=False)
                    self.latest[win_name] = (frame, targets)
                except queue.Empty:
                    break

            # 2. Display frames and process targets
            for win_name, (frame, targets) in self.latest.items():
                cv2.imshow(win_name, frame)

                # Here’s where you can act on the targets:
                if targets is not None and len(targets) > 0:
                    # print(f"[{win_name}] detected targets:", targets)
                    self.add_to_target_list(targets, MIN_DISTANCE)
            
            # 3. Manage gun logic
            self.check_emergency_targets()
            for gun in self.guns:
                if gun.is_free():
                    self.pop_optimized(gun)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        for thread in threading.enumerate():
            print(f"Thread {thread.name} is alive: {thread.is_alive()}")
        print("Exiting...")
        exit(0)

if __name__ == "__main__":
    gun_info = [((100.0, 95.0), 0)]  # example
    # cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1]), (CAMERA_INDEX_2, CAMERA_LOCATION_2, homography_matrices[2])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    # cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0])]
    try:
        # brain = Brain(gun_info, cam_info)
        # brain.game_loop_independent()
        Brain(gun_info, cam_info).game_loop_display()
    except KeyboardInterrupt:
        for thread in threading.enumerate():
            print(f"Thread {thread.name} is alive: {thread.is_alive()}")
        print("Exiting...")
        exit(0)



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


#     def add_to_target_list(
#         self, new_targets: list, camera: Eye, distance: float = MIN_DISTANCE
#     ):
#         if new_targets is None:
#             return
#         for target in new_targets:
#             # real_coords = self.calculate_real_coords(target, camera)
#             if not self.too_close(target, distance):
#                 self.add_smart(target, camera)

#     def add_smart(self, target, cam):
#         # this is temporary:
#         self.targets.append(target)

#     def calculate_real_coords(self, target, camera):
#         # for now, we are assuming that the homography matrix is built such that the real coordinates are returned originally.
#         return target

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
#         SAMPLE_RATE = 5
#         self.running = True

#         def show_display(camera_index: int):
#             print("display thread started")
#             eye = self.eyes[camera_index]
#             while self.running:
#                 frame = eye.get_frame()
#                 if frame is not None:
#                     eye.raw_handler.add(frame)
#                     eye.raw_handler.display()

#         def run_gun():
#             gun = self.guns[0]
#             print(f"Gun {gun.gun_location} is ready to shoot")
#             while self.running:
#                 if self.targets:
#                     target = self.targets.pop(0)
#                     print(f"Gun {gun.gun_location} is aiming at target {target}")
#                     angle = self.calculate_angle(target)
#                     print("angle to shoot:", angle)
#                     gun.rotate(angle * -1)
#                     gun.shoot()
#                     time.sleep(0.5)

#         # Start threads
#         display_1 = threading.Thread(target=show_display, args=(0,), daemon=True)
#         display_1.start()

#         # gun_thread = threading.Thread(target=run_gun, daemon=True)
#         # gun_thread.start()

#         while True:
#             self.timestep += 1

#             to_check = self.timestep % 15 == 14
#             to_init = self.timestep == 5

#             if self.timestep % SAMPLE_RATE == 0:
#                 for eye in self.eyes:
#                     frame = eye.get_frame()
#                     if frame is not None:
#                         targets = eye.add(frame, to_check, to_init)
#                         self.add_to_target_list(targets, eye, MIN_DISTANCE)

#             if to_check or to_init:
#                 print("targets in brain:", self.targets)

#             if cv2.waitKey(1) == 27:
#                 self.running = False
#                 break

#         cv2.destroyAllWindows()
    



