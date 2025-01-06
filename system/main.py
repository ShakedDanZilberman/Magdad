import time
import threading

from import_defence import ImportDefence

with ImportDefence():
    import cv2
    import numpy as np
    from pyfirmata import Arduino, util
    import matplotlib.pyplot as plt

from contours import ContoursHandler
from changes import ChangesHandler
from cameraIO import detectCameras
from image_processing import RawHandler, ImageParse
from mouseCamera import MouseCameraHandler
from object_finder import show_targets, get_targets
from motion import DifferenceHandler
from laser import LaserPointer
from cameraIO import Camera
from object_finder import average_of_heatmaps
from gui import LIDARDistancesGraph

timestep = 0
centers = [(30, 60)]

CAMERA_INDEX = 1

DIFF_THRESH = 0
INITIAL_CONTOUR_EXTRACT_FRAME_NUM = 30
CHECK_FOR_NEW_OBJECTS = 48


def shoot(target):
    print("Shooting", target)


def laser_thread():
    print("Laser thread started")
    global centers, laser_point
    laser_pointer = LaserPointer()
    # graph = LIDARDistancesGraph()
    previous_distances = [0] * 3
    while True:
        my_centers = centers.copy()
        my_centers = sorted(my_centers, key=lambda x: x[0])
        for center in my_centers:
            laser_pointer.move(center)
            # laser_pointer.move((300, 190))

            distance = laser_pointer.distance()
            previous_distances.append(distance)
            previous_distances.pop(0)
            measured_distance = sum(previous_distances) / 3
            print("Measured distance:", measured_distance)
            # graph.add_distance(measured_distance)
            # graph.plot()

            time.sleep(0.1)
        time.sleep(0.1)

    plt.ioff()
    plt.show()




def hit_cursor_main():
    global CAMERA_INDEX, timestep, centers
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    handler = MouseCameraHandler()
    laser = threading.Thread(target=laser_thread)
    laser.start()

    title = handler.TITLE

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, handler.mouse_callback)

    while True:
        img = cam.read()

        handler.add(img)
        handler.display()
        mousePos = handler.getMousePosition()
        centers = [mousePos]
        
        if cv2.waitKey(1) == 32:  # Whitespace
            shoot(mousePos)

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()



def main():
    global CAMERA_INDEX, timestep, centers
    detectCameras()
    cam = Camera(CAMERA_INDEX)
    rawHandler = RawHandler()
    changesHandler = ChangesHandler()
    differenceHandler = DifferenceHandler()
    contoursHandler = ContoursHandler()
    laser = threading.Thread(target=laser_thread)
    laser.start()
    target_queue = []
    target = None

    
    number_of_frames = 0
    while True:
        number_of_frames += 1
        img = cam.read()

        for handler in [
            rawHandler,
            changesHandler,
            differenceHandler,
            contoursHandler,
        ]:
            handler.add(img)
            # handler.display()
        rawHandler.display()
        # changesHandler.display()
        img_contours = contoursHandler.get()
        img_changes = changesHandler.get()
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
        if number_of_frames == INITIAL_CONTOUR_EXTRACT_FRAME_NUM:
            for center in centers_contours:
                target_queue.append(center)
                centers.append(center)
        elif number_of_frames%CHECK_FOR_NEW_OBJECTS == CHECK_FOR_NEW_OBJECTS-1 and ImageParse.image_sum(differenceHandler.get()) <= DIFF_THRESH:
                for center in centers_changes:
                    centers.append(center)
                    target_queue.append(center)
                changesHandler.clear()
                img_changes = changesHandler.get(img)
        # print("queue:", centers)
        if number_of_frames % CHECK_FOR_NEW_OBJECTS == 0 and len(target_queue) > 0:
            target = target_queue.pop(0)
            
            if not target == None:
                centers.append(target)
                target = None
        
        if cv2.waitKey(1) == 32:  # Whitespace
            changesHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hit_cursor_main()
