import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from pyfirmata import Arduino, util
from time import sleep


CAMERA_INDEX = 1
WINDOW_NAME = 'Camera Connection'
MAX_CAMERAS = 10

# Set up the Arduino board (replace 'COM8' with your Arduino's COM port)
board = Arduino('COM8')  # Adjust the COM port based on your system

# Define the pin for the servo (usually PWM pins)
servoV_pin = 5
servoH_pin = 3# Servo control pin (could be any PWM pin)
laser_pin = 8
board.digital[laser_pin].write(1)
# Attach the servo to the board
servoV = board.get_pin(f'd:{servoV_pin}:s')  # 's' means it's a servo
servoH = board.get_pin(f'd:{servoH_pin}:s')
# Start an iterator thread to read analog inputs
it = util.Iterator(board)
it.start()
servoH.write(9)
servoV.write(9)
sleep(1)


A0 = 0
B0 = 0
C0 = -0.12
D0 = 76

A1 = 0
B1 = 0
C1 = -0.15
D1 = 73
# Main loop to control the servo
def angle_calc(coordinates):
    X = coordinates[0]
    Y = coordinates[1]
    angleX = D0 + C0*X +B0*X**2 + A0*X**3
    angleY = D1 + C1*Y +B1*Y**2 + A1*Y**3
    return angleX, angleY

mx,my=0,0

def click_event(event, x, y, flags, param):
    global mx,my
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked coordinates: {relative_x}, {relative_y}")
        mx,my=x,y
        

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
    lower_red1 = np.array([0, 120, 70])   # Lower range for red
    upper_red1 = np.array([10, 255, 255]) # Upper range for red
    lower_red2 = np.array([170, 120, 70]) # Lower range for red (wrapping around 180 degrees)
    upper_red2 = np.array([180, 255, 255])# Upper range for red
    
    # Create masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # Combine masks

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return (0,0)  # No red point found

    # Assume the largest contour is the red point (adjust as needed)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the center of the red point
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return (0,0)  # Avoid division by zero

    cX = int(M["m10"] / M["m00"])  # x-coordinate
    cY = int(M["m01"] / M["m00"])  # y-coordinate

    return cX, cY
        

def main():
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)
    ret_val, img = cam.read()
    global mx,my

    cv2.setMouseCallback(WINDOW_NAME, click_event)
    while True:
        ret_val, img = cam.read()
        (rx,ry)=find_red_point(img)
        cv2.circle(img,(rx,ry),10,(0,0,255),-1)
        cv2.circle(img,(mx,my),10,(255,0,0),-1)

        # Press Escape or close the window to exit
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        

        # angleX, angleY = angle_calc([mx-rx,my-ry])
        #magic numbers!!!
        angleX, angleY = angle_calc([mx,my])
        print(angleX,angleY)
        servoH.write(angleX)
        servoV.write(angleY)
        sleep(0.1)
        cv2.imshow(WINDOW_NAME, img)



    cv2.destroyAllWindows()
    board.exit()


if __name__ == '__main__':
    main()




