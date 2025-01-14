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

timestep = 0  # Global timestep, used to keep track of the number of frames processed
laser_targets = [(30, 60)]  # List of targets for the laser pointer, used to share information between threads

CAMERA_INDEX = 1

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
    """
    global CAMERA_INDEX, timestep, laser_targets
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    handler = MouseCameraHandler()
    laser = threading.Thread(target=laser_thread)
    laser.start()

    cv2.namedWindow(handler.TITLE)
    cv2.setMouseCallback(handler.TITLE, handler.mouse_callback)

    while True:
        img = cam.read()

        handler.add(img)
        handler.display()

        mousePos = handler.getMousePosition()
        laser_targets = [mousePos]
        
        if cv2.waitKey(1) == 32:  # Whitespace
            shoot(mousePos)

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
        circles_high, circles_low, centers_contours = get_targets(img_contours)
        targets_contours = circles_high, circles_low, centers_contours
        circles_high, circles_low, centers_changes = show_targets("img_contours", img_contours, targets_contours)

        if isinstance(img_changes, np.ndarray) and img_changes.size > 1:
            circles_high, circles_low, centers_changes = get_targets(img_changes)
            targets_changes = circles_high, circles_low, centers_changes
            circles_high, circles_low, centers_changes = show_targets("img_changes", img_changes, targets_changes)
        else:
            circles_high, circles_low, centers_changes = [], [], []
            targets_changes = circles_high, circles_low, centers_changes
        if timestep == INITIAL_CONTOUR_EXTRACT_FRAME_NUM:
            for center in centers_contours:
                target_queue.append(center)
                laser_targets.append(center)
        elif timestep%CHECK_FOR_NEW_OBJECTS == CHECK_FOR_NEW_OBJECTS-1 and ImageParse.image_sum(differenceHandler.get()) <= DIFF_THRESH:
                for center in centers_changes:
                    laser_targets.append(center)
                    target_queue.append(center)
                changesHandler.clear()
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


if __name__ == "__main__":
    main()
