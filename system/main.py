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
from image_processing import RawHandler
from object_finder import show_targets, get_targets
from motion import DifferenceHandler
from laser import LaserPointer
from cameraIO import Camera
from object_finder import average_of_heatmaps

timestep = 0
centers = [[90, 90]]

IMG_WIDTH = 240
IMG_HEIGHT = 320

CAMERA_INDEX = 1


def laser_thread():
    print("Laser thread started")
    global centers, laser_point
    laser_pointer = LaserPointer()
    while True:
        my_centers = centers.copy()
        my_centers = sorted(my_centers, key=lambda x: x[0])
        for center in my_centers:
            laser_pointer.move(center)
            laser_point = center
            time.sleep(0.3)
        time.sleep(0.2)


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
            handler.display(img)
        
        # changes_heat_map = newPixelsHandler.get()

        average = average_of_heatmaps(changesHandler.get(img), contoursHandler.get())
        cv2.imshow("changesHandler.get()", changesHandler.get(img))
        cv2.imshow("Average of Heatmaps", average)
        circles_high, circles_low, centers = get_targets(average)
        show_targets(img, (circles_high, circles_low, centers))
        
        if cv2.waitKey(1) == 32:  # Whitespace
            changesHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
