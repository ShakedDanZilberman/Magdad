from eyefile import Eye
from gun import Gun, DummyGun
from cameraIO import Camera
import threading
from constants import *
import time
import numpy as np
import queue
from serial.tools import list_ports as coms
from image_processing import ImageParse
# from vis_production import CameraProducer

import openvino
import cv2


class CameraProducer(threading.Thread):
    def __init__(self, camera_index, eye, output_queue):
        super().__init__(daemon=True)
        self.eye = eye
        self.output_queue = output_queue
        self.win_name = f"Cam {camera_index}"
        self.timestep = 0

    def run(self):
        while True:
            self.timestep+=1
            frame = self.eye.camera.read(self.timestep)
            # Run your add_yolo processing (which returns targets)
            if self.timestep % 15 == 14:
                print(f"checking frame for targets")    
                targets = self.eye.add_yolo(frame)
            else:
                targets = self.eye.add_yolo(frame, False)
            # Grab the visualized image (after display())
            self.eye.yolo_handler.prepare_to_show()
            vis = self.eye.yolo_handler.get_vis()
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
    def __init__(self, guns, cam_info):
        """
        cam info is an array of tuples, for each camera: (cam_index, CAMERA_LOCATION_0)
        gun_locations is an array of tuples, for each gun: its location
        """
        self.guns = guns
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
        target_x, target_y = target
        # print("tx, ty:", (tx, ty))
        list_targets = list(self.targets.keys())
        list_history = list(self.history.keys())
        if isinstance(target_x, np.ndarray) or isinstance(target_y, list):
            print(target_x, target_y)
        try:
            target_x = float(target_x)
            target_y = float(target_y)
        except TypeError:
            print("tx, ty are not float: ", (target_x, target_y))
            return True
        for (other_x, other_y) in list_targets:
            # print("ox, oy:", (ox, oy))
            if (target_x - other_x)**2 + (target_y - other_y)**2 <= min_dist**2:
                # print("in too_close: target is too close to existing target")
                return True
        for (other_x, other_y) in list_history:
            if (target_x - other_x)**2 + (target_y - other_y)**2 <= min_dist**2:
                return True
        # print("list targets: ", list_targets)
        # print("list history: ", list_history)
        return False

    def add_to_target_list(self, new_targets, distance: float = MIN_DISTANCE):
        # print("new targets: ", new_targets)
        if len(new_targets) == 0:
            return
        for target in new_targets:
            # print("target: ", target, "distance: ", distance)
            is_too_close = self.too_close(target, distance)
            if not is_too_close:
                # print("in add_to_target_list: adding target: ", target)
                self.add_smart(target)

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
        return angle

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
                self.add_to_target_list(targets, 0)

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
                target = emergency_targets.pop(0)
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
            print(f"pop_optimized: history is {self.history}")
            return
        for location, priority in self.targets.items():
            if gun.target_stack is not None and len(gun.target_stack) > 0:
                angular_dist = self.angle_diff(gun.target_stack[0][0], location, gun.gun_index)
                if angular_dist < min_anglular_distance:
                    min_anglular_distance = angular_dist
                    closest_target = (location, priority)
                    # print(f"pop_optimized: closest target: {closest_target}, angular distance: {angular_dist}")
                    # print(f"pop_optimized: gun location: {gun.gun_location}, target: {location}, angular distance: {angular_dist}")
        if len(gun.target_stack) > 2:
            return
        if closest_target is not None:
            # print("closest target: ", closest_target)
            priority = self.targets.pop(closest_target[0])
            # print(f"targets in brain after pop: {self.targets}")
            gun.target_stack.append((closest_target[0], priority))
            print(f"assigned target {closest_target} to gun {gun.gun_index}")
            self.history[closest_target[0]] = (priority, self.timestep)
            print(f"pop_optimized: history is {self.history}")
        return

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
        to_delete = []
        for location, data in self.history.items():
            if np.abs(data[1] - self.timestep) > HISTORY_DELAY:
                to_delete.append(location)
        for deleted in to_delete:
            if deleted in self.history:
                self.history.pop(deleted)
                print(f"deleted target {deleted} from history")
                # print("history: ", self.history)
                # print("targets: ", self.targets)
                # print("history after pop: ", self.history)

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
                    print("angle to shoot: ", angle)
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
            # self.check_emergency_targets()
            # print(f"targets in brain: {self.targets}")
            for gun in self.guns:
                
                if gun.is_free():
                    # assign the target to the gun
                    self.pop_optimized(gun)
            
            self.update_history()
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
            print(f"in gun thread: Gun {gun.gun_location} is ready to shoot")
            while True:
                if gun.target_stack is not None and len(gun.target_stack)>0:
                    # print("gun target stack: ", gun.target_stack)
                    # Get the target coordinates from the last camera
                    target = gun.target_stack[0]            
                    # print(f"Gun {gun.gun_location} is aiming at target {target}")
                    # Calculate the angle to rotate to
                    print(f"gun {gun.gun_location} target stack: {gun.target_stack}")
                    angle = self.calculate_angle_from_gun(target[0], gun_index)
                    print(f"gun {gun.gun_location} - angle to shoot: ", angle)
                    gun.rotate(angle)
                    time.sleep(0.1)
                    gun.shoot()
                    time.sleep(0.3)
                    gun.rotate(0)
                    print("gun {gun.gun_location} going to zero 1...")
                    print("gun {gun.gun_location} going to zero 2...")
                    print("gun {gun.gun_location} going to zero 3...")
                    time.sleep(6)
                    print("shot fired")
                    gun.target_stack.pop(0)
                    # print("gun target stack after pop: ", gun.target_stack)
                    print("in gun thread: sleeping for 1 second")
                    time.sleep(0.1)
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
            frames = []            
            # 2. Display frames and process targets
            for win_name, (frame, targets) in self.latest.items():
                frames.append(ImageParse.resize_proportionally(frame, 0.5))
                # cv2.imshow(win_name, frame)
                # Here’s where you can act on the targets:
                if targets is not None and len(targets) > 0:
                    # print(f"in main thread: [{win_name}] detected targets:", targets)
                    self.add_to_target_list(targets, MIN_DISTANCE)
            if len(frames) > 0:
                combined_frame = np.hstack(frames)
                cv2.imshow("Combined", combined_frame)
                
            
            # 3. Manage gun logic
            self.check_emergency_targets()
            for gun in self.guns:
                if gun.is_free():
                    self.pop_optimized(gun)
            if cv2.waitKey(1) == 27:
                break

            self.update_history()
                # print("history: ", self.history)
                # print("targets: ", self.targets)
                # print("history after pop: ", self.history)
        cv2.destroyAllWindows()
        print("Please kill the terminal.")
        print("just kill it...")
        print("dont let it suffer")
        print("there is good pain, and there is meangingless pain")
        print("this is the latter")
        print("also, escape to exit")
        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                thread.join()
                print(f"Thread {thread.name} is alive: {thread.is_alive()}")
        print("Exiting...")
        exit(0)



if __name__ == "__main__":
    white_gun = Gun(gun_location=(195.5,100.0),
                    index=0,
                    COM="COM14", 
                    print_flag=False)
    
    black_gun = Gun(gun_location=(100.0,95.0),
                    index=1,
                    COM="COM18", 
                    print_flag=False)
    
    # gun_info = [((195.5,100.0), 0, "COM14", False), ((100.0,95.0), 1, "COM18", False)]
    guns = [white_gun, black_gun]
    

    from Brain import Brain
    # guns = []
    # cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    from constants import *
    import threading
    cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), 
                (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1]), 
                (CAMERA_INDEX_2, CAMERA_LOCATION_2, homography_matrices[2])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    # cam_info = [(CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1])]
    try:
        brain = Brain(guns, cam_info)
        brain.game_loop_independent()
        # brain.game_loop_display()
    except KeyboardInterrupt:
        for thread in threading.enumerate():
            print(f"Thread {thread.name} is alive: {thread.is_alive()}")
        print("Exiting...")
        exit(0)




# in gun thread: sleeping for 1 second
# assigned target ((np.float32(220.61852), np.float32(6.505526)), 1) to gun 0
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(220.76598), np.float32(6.533121)), 1) to gun 1
# gun (195.5, 100.0) target stack: [((np.float32(220.61852), np.float32(6.505526)), 1), ((np.float32(220.61852), np.float32(6.505526)), 1)]
# angle to shoot:  15.038155
# Rotating to 15.038154602050781 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(220.61852), np.float32(6.505526)), 1) to gun 0
# gun (100.0, 95.0) target stack: [((np.float32(220.61852), np.float32(6.505526)), 1), ((np.float32(220.76598), np.float32(6.533121)), 1)]
# angle to shoot:  53.733536
# Rotating to 53.73353576660156 degrees
# in gun rotate: sleep 0.5 secs
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(220.61852), np.float32(6.505526)), 1) to gun 1
# gun (195.5, 100.0) target stack: [((np.float32(220.61852), np.float32(6.505526)), 1), ((np.float32(220.61852), np.float32(6.505526)), 1)]
# angle to shoot:  15.038155
# Rotating to 15.038154602050781 degrees
# in gun rotate: sleep 0.5 secs
# gun (100.0, 95.0) target stack: [((np.float32(220.76598), np.float32(6.533121)), 1), ((np.float32(220.61852), np.float32(6.505526)), 1)]
# angle to shoot:  53.775433
# Rotating to 53.77543258666992 degrees
# shot fired
# in gun thread: sleeping for 1 second
# assigned target ((np.float32(220.61852), np.float32(6.505526)), 1) to gun 0
# in gun rotate: sleep 0.5 secs
# shot fired