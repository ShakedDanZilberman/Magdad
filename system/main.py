import time
import threading

from import_defence import ImportDefence

with ImportDefence():
    import cv2
    import numpy as np
    from pyfirmata import Arduino, util
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

from contours import ContoursHandler
from changes import ChangesHandler
from cameraIO import detectCameras
from image_processing import RawHandler, ImageParse
from mouseCamera import MouseCameraHandler
from object_finder import show_targets, get_targets
from motion import DifferenceHandler
from yolo import YOLOHandler
from laser import LaserPointer
from cameraIO import Camera
from object_finder import average_of_heatmaps
from gui import LIDARDistancesGraph
from gun import Gun, DummyGun
from constants import CAMERA_INDEX
from object_finder import Targets

timestep = 0  # Global timestep, used to keep track of the number of frames processed
laser_targets = [(30, 60)]  # List of targets for the laser pointer, used to share information between threads
gun_targets = []  # List of targets for the gun, used to share information between threads
P_ERROR = -75
I_ERROR = -25
D_ERROR = 0
total_error = 0
last_error = 0
error = 0
fix = 0
DIFF_THRESH = 0
INITIAL_CONTOUR_EXTRACT_FRAME_NUM = 30
CHECK_FOR_NEW_OBJECTS = 48


def shoot(target):
    print("Shooting", target)


def laser_thread():
    """
    Thread that moves the laser pointer to the target.
    The targets are aquired as an asynchronous input from the main thread.
    Only this thread may access LaserPointer which controls the laser pointer and the LIDAR.
    """
    print("Laser thread started.")
    global laser_targets, laser_point
    laser_pointer = LaserPointer()
    print("Laser pointer initialised and connected.")
    distances_interative_graph = LIDARDistancesGraph(show=False)
    while True:
        my_targets = laser_targets.copy()
        my_targets = sorted(my_targets, key=lambda x: x[0])  # sort by x coordinate to create a smooth left-to-right movement
        for center in my_targets:
            # Move the laser pointer to the target
            laser_pointer.move(center)

            # Measure the distance to the target
            distance = laser_pointer.distance()
            distances_interative_graph.add_distance(distance)
            print("Measured distance:", distances_interative_graph.distance())
            distances_interative_graph.plot()

            time.sleep(0.1)
        time.sleep(0.1)

    plt.ioff()
    plt.show()


def hit_cursor_main():
    """
    An alternative to the main function that uses the mouse cursor as the target for the laser pointer.
    It does not use any image processing to detect targets.
    """
    global CAMERA_INDEX, timestep, laser_targets
    import fit
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    handler = MouseCameraHandler()
    # laser = threading.Thread(target=laser_thread)
    # laser.start()  # comment this line to disable the laser pointer
    gun = Gun()  # DummyGun() or Gun()

    cv2.namedWindow(handler.TITLE)
    cv2.setMouseCallback(handler.TITLE, handler.mouse_callback)
    thetaX = 90.00
    while True:
        img = cam.read()

        handler.add(img)
        handler.display()

        mousePos = handler.getMousePosition()
        laser_targets = [mousePos]
        last_thetaX = thetaX
        thetaX, expected_volt = fit.bilerp(*mousePos)
        # use PID
        global fix
        if len(thetaX) == 2:
            thetaX = thetaX[0]
        if thetaX != last_thetaX:
            gun.rotate(thetaX)
            global last_error, total_error
            last_error = 0 
            total_error = 0
            time.sleep(0.5)
        else:
            gun.rotate(thetaX + fix)
        motor_volt = gun.get_voltage()
        fix = PID(expected_volt,motor_volt) 
        
        
        if cv2.waitKey(1) == 32:  # Whitespace
            gun.shoot()
        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def PID(expected_volt, motor_volt):
    # use PID      
        global total_error, last_error, error
        last_error = error
        error = motor_volt - expected_volt
        dif_error = error - last_error
        total_error += error
        print("Motor voltage:", motor_volt, "Expected voltage:", expected_volt, "Error:", error)
        return error*P_ERROR + total_error*I_ERROR + dif_error*D_ERROR

def just_changes_main():
    """
    This function uses the changes in the image to detect targets.
    It does not use contours or YOLO.
    It is a simplified version of the main function.
    """
    global CAMERA_INDEX, timestep, gun_targets
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    rawHandler = RawHandler("Whitespace to clear")
    changesHandler = ChangesHandler()

    def gun_thread():
        """
        Thread that moves the gun to the target and shoots.
        The targets are aquired as an asynchronous input from the main thread.
        """
        import fit
        print("Gun thread started.")
        global gun_targets
        gun = Gun()
        print("Gun initialised and connected.")
        while True:
            my_targets = gun_targets.copy()
            # Sort by x coordinate to create a smooth right-to-left movement
            my_targets = sorted(my_targets, key=lambda x: x[0], reverse=True)
            for center in my_targets:
                # Move the laser pointer to the target
                thetaX, expected_volt = fit.bilerp(*center)
                # use PID
                motor_volt = gun.get_voltage()
                error = PID(expected_volt,motor_volt)  
                gun.rotate(thetaX + error)
                print("Shooting", center)
            time.sleep(1)
        # TODO: It seems gun_thread is not moving, even though the targets are being updated. FIX

    
    gun = threading.Thread(target=gun_thread)
    gun.start()
    
    while True:
        timestep += 1
        img = cam.read()

        rawHandler.add(img)
        rawHandler.display()
        
        changesHandler.add(img)
        changesHandler.display()

        img_changes = changesHandler.get()

        FRAME_TITLE = "Targets from Changes"

        if isinstance(img_changes, np.ndarray) and img_changes.size > 1:
            targets_changes = circles_high, circles_low, centers_changes = get_targets(img_changes)
            show_targets(FRAME_TITLE, img_changes, targets_changes)
        else:
            targets_changes = circles_high, circles_low, centers_changes = [], [], []
            show_targets(FRAME_TITLE, img_changes, targets_changes)

        # add the targets from the changes to the queue
        if len(centers_changes) > 0:
            # Remove from centers_changes any targets that are less than 30 pixels apart (unique targets)
            targets = []
            pixel_distance = 30
            for center in centers_changes:
                if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
                    targets.append(center)
            gun_targets = targets.copy()
            print("Targets:", gun_targets)
        
        if cv2.waitKey(1) == 32:  # Whitespace
            changesHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def main():
    global CAMERA_INDEX, timestep, laser_targets
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    rawHandler = RawHandler()
    changesHandler = ChangesHandler()
    differenceHandler = DifferenceHandler()
    contoursHandler = ContoursHandler()
    yoloHandler = YOLOHandler()
    laser = threading.Thread(target=laser_thread)
    laser.start()
    target_queue = []
    target = None
    
    while True:
        timestep += 1
        img = cam.read()

        for handler in [
            rawHandler,
            changesHandler,
            differenceHandler,
            contoursHandler,
            yoloHandler,
        ]:
            handler.add(img)
            handler.display()
        # rawHandler.display()
        # changesHandler.display()
        img_contours = contoursHandler.get()
        img_changes = changesHandler.get()
        img_objects = yoloHandler.get()  # TODO: Use the bounding boxes to get the targets
        # TODO: make the decision algorithm more robust, clear, and DRY
        # get the targets using the contour detection method
        circles_high, circles_low, centers_contours = get_targets(img_contours)
        targets_contours = circles_high, circles_low, centers_contours
        # show them on screen
        circles_high, circles_low, centers_changes = show_targets("img_contours", img_contours, targets_contours)

        if isinstance(img_changes, np.ndarray) and img_changes.size > 1:
            # get the targets using the image diffrecnce method
            circles_high, circles_low, centers_changes = get_targets(img_changes)
            targets_changes = circles_high, circles_low, centers_changes
            # show them on screen
            circles_high, circles_low, centers_changes = show_targets("img_changes", img_changes, targets_changes)
        else:
            circles_high, circles_low, centers_changes = [], [], []
            targets_changes = circles_high, circles_low, centers_changes
        # extract the objects using contour detection
        if timestep == INITIAL_CONTOUR_EXTRACT_FRAME_NUM:
            for center in centers_contours:
                target_queue.append(center)
                laser_targets.append(center)
        # at a constant rate, check for changes between the current image and and the original image
        elif timestep%CHECK_FOR_NEW_OBJECTS == CHECK_FOR_NEW_OBJECTS-1 and ImageParse.image_sum(differenceHandler.get()) is not None and ImageParse.image_sum(differenceHandler.get()) <= DIFF_THRESH:
                # add all new objects in the image 
                for center in centers_changes:
                    laser_targets.append(center)
                    target_queue.append(center)
                # immediately after adding targets to the target queue, reset the image changes heatmap, 
                # so that the movements of objects caused by shooting do not count as new targets
                changesHandler.clear()
                # get the changes image again
                img_changes = changesHandler.get()
        # print("queue:", centers)
        if timestep % CHECK_FOR_NEW_OBJECTS == 0 and len(target_queue) > 0:
            target = target_queue.pop(0)
            
            if not target == None:
                laser_targets.append(target)
                target = None
        
        if cv2.waitKey(1) == 32:  # Whitespace
            changesHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def gun_thread(target_manager):
        """
        Thread that moves the gun to the target and shoots.
        The targets are aquired as an asynchronous input from the main thread.
        """
        print("Gun thread started.")
        global gun_targets
        gun = Gun(print_flag=True)
        while True:
            center = target_manager.pop()
            if center is not None:
                gun.aim_and_fire_target(center)
                print("Shooting", center)
                time.sleep(1)
                target_manager.clear()
    

def main_using_targets():
    global CAMERA_INDEX, timestep, gun_targets
    from gui import GUI
    from constants import useJustYOLO
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    rawHandler = RawHandler()
    target_manager = Targets()
    gui = GUI()
    
    gun = threading.Thread(target=gun_thread, args=(target_manager,))
    gun.start()
    
    while True:
        timestep += 1
        img = cam.read()
        rawHandler.add(img)
        if useJustYOLO:
            target_manager.addJustYOLO(timestep, img)
        else:
            target_manager.add(timestep, img)
        gui.add(img, target_manager.target_queue, target_manager.changes_handler.get(), target_manager.contours_handler.get(),
                target_manager.low_targets, target_manager.high_targets, target_manager.yolo_handler.get(), target_manager.yolo_centers)
        gui.display()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def test_main():
    global CAMERA_INDEX, timestep, gun_targets
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    rawHandler = RawHandler()
    contours_handler = ChangesHandler()

    while True:
        timestep += 1
        img = cam.read()
        rawHandler.add(img)
        rawHandler.display()
        contours_handler.add(img)
        contours_handler.display()
        img_contours = contours_handler.get()
        targets_contours = _, _, contours_centers = get_targets(img_contours)
        show_targets("targets from contours", img_contours, targets_contours)
        
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def test():
    gun = Gun(print_flag=True)
    while True:
        gun.rotate(60)
        print(gun.get_voltage()) 
        gun.shoot()
        time.sleep(1)
    print("done")


if __name__ == "__main__":
    #test()
    hit_cursor_main()
    #just_changes_main()
    # main_using_targets()
