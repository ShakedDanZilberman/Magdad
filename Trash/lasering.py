import cv2
import numpy as np
import math
import time
from pyfirmata import Arduino, util
from time import sleep
import system.fit as fit
import system.pid

CAMERA_INDEX = 1 

WINDOW_NAME = 'Camera Connection'

# Set up the Arduino board (replace 'COM8' with your Arduino's COM port)
board = Arduino('COM8')  # Adjust the COM port based on your system

# Define the pin for the servo (usually PWM pins)
servoV_pin = 5 # Servo control pin (could be any PWM pin)
servoH_pin = 3
laser_pin = 8 # laser pin
board.digital[laser_pin].write(1) # turn on laser
# Attach the servo to the board, V for vertical servo and H for horizontal servo
servoV = board.get_pin(f'd:{servoV_pin}:s')  # 's' means it's a servo
servoH = board.get_pin(f'd:{servoH_pin}:s')
# Start an iterator thread to read analog inputs
it = util.Iterator(board)
it.start()



coeffsX, coeffsY = fit.get_coeefs()
def angle_calc(x,y):
    global coeffsX, coeffsY
    angleX = fit.evaluate_polynomial(x,y,coeffsX)
    angleY = fit.evaluate_polynomial(x,y,coeffsY)
    return angleX, angleY


def find_red_point(frame):
    """
    Finds the (x, y) coordinates of the single red point in the image.

    Args:
        frame: The input image (BGR format).

    Returns:
        A tuple (x, y) representing the center of the red point if found, or None if no red point is detected.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red in HSV
    lower_red1 = np.array([0, 120, 70])  # Lower range for red
    upper_red1 = np.array([10, 255, 255])  # Upper range for red
    lower_red2 = np.array([170, 120, 70])  # Lower range for red (wrapping around 180 degrees)
    upper_red2 = np.array([180, 255, 255])  # Upper range for red

    # Create masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # Combine masks

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return (0, 0)  # No red point found

    # Assume the largest contour is the red point (adjust as needed)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the center of the red point
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return (0, 0)  # Avoid division by zero

    cX = int(M["m10"] / M["m00"])  # x-coordinate
    cY = int(M["m01"] / M["m00"])  # y-coordinate

    return cX, cY

mouse_x, mouse_y = 0, 0
def click_event(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y


def main():
    global mouse_x, mouse_y

    #TODO: move to function
    # initilization
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)
    ret_val, img = cam.read()
    cv2.setMouseCallback(WINDOW_NAME, click_event)
    cv2.imshow(WINDOW_NAME, img)

    # main loop
    while True:
        # read image
        ret_val, img = cam.read()
        # display circles for laser and mouse
        (laser_x, laser_y) = find_red_point(img)
        cv2.circle(img, (laser_x, laser_y), 7, (0, 0, 255), -1)
        cv2.circle(img, (mouse_x, mouse_y), 7, (255, 0, 0), -1)

        angleX, angleY = angle_calc([mouse_x, mouse_y])
        print(angleX, angleY)
        servoH.write(angleX)
        servoV.write(angleY)
        sleep(0.1) 
        cv2.imshow(WINDOW_NAME, img)


        # Press Escape or close the window to exit
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
       
   
    cv2.destroyAllWindows()
    board.exit()


if __name__ == '__main__':
    main()




