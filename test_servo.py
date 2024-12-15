import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from pyfirmata import Arduino, util
from time import sleep, time

# Configuration for the Arduino and Camera
CAMERA_INDEX = 1
WINDOW_NAME = 'Camera Connection'

board = Arduino('COM8')  # Adjust the COM port
servoV_pin = 4
servoH_pin = 3
laser_pin = 8

clicked = False
board.digital[laser_pin].write(1)
servoV = board.get_pin(f'd:{servoV_pin}:s')
servoH = board.get_pin(f'd:{servoH_pin}:s')
it = util.Iterator(board)
it.start()

servoH.write(90)
servoV.write(90)
sleep(1)

# Calibration Coefficients
A0 = 0
B0 = 0
C0 = 0.5
D0 = 90
A1 = 0
B1 = 0
C1 = 0.5
D1 = 90

mx, my = 0, 0
time_stamps = []  # To store timestamps
mx_values = []  # To store mx values
my_values = []  # To store my values
angleX_values = []  # To store angleX values
angleY_values = []  # To store angleY values
angleX = 90
angleY = 90
def click_event(event, x, y, flags, param):
    global mx, my
    if event == cv2.EVENT_LBUTTONDOWN:
        mx, my = x, y
        global clicked
        clicked = True
def find_red_point(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (0, 0)

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return (0, 0)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def remove_consecutive_duplicates(input_list):
    if not input_list:  # If the list is empty, return an empty list
        return []

    result = [input_list[0]]  # Start with the first element in the list
    for value in input_list[1:]:
        if value != result[-1]:  # Add the value if it's different from the last added
            result.append(value)
    return result
def main():
    global mx, my, angleX, angleY, clicked

    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cv2.setMouseCallback(WINDOW_NAME, click_event)

    start_time = time()  # Record the start time
    while True:
        ret_val, img = cam.read()
        if not ret_val:
            break

        (rx, ry) = find_red_point(img)
        cv2.circle(img, (rx, ry), 10, (0, 0, 255), -1)
        cv2.circle(img, (mx, my), 10, (255, 0, 0), -1)

        # Calculate servo angles
        if clicked:
            print("Clicked")
            angleX = 90+random.randint(-20,20)
            angleY = 90+random.randint(-20,20)
            clicked = False

        servoH.write(angleX)
        servoV.write(angleY)
        sleep(0.1)

        # Store data for graphs
        current_time = time() - start_time
        time_stamps.append(current_time)
        mx_values.append(mx)
        my_values.append(my)
        angleX_values.append(angleX)
        angleY_values.append(angleY)

        cv2.imshow(WINDOW_NAME, img)

        # Exit conditions
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    print("hi")
    cam.release()
    cv2.destroyAllWindows()
    board.exit()

    mx_values.pop()
    my_values.pop()
    angleX_values.pop(0)
    angleY_values.pop(0)
    # Plot the graphs
    # Graph 1: mx vs. angleX
    print(mx_values, angleX_values)
    plt.figure(figsize=(10, 5))
    plt.plot(mx_values, angleX_values, label='angleX vs mx', color='green')
    plt.xlabel('mx (X coordinate)')
    plt.ylabel('angleX (Servo Angle)')
    plt.title('Servo Angle X vs Mouse X Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graph 2: my vs. angleY
    plt.figure(figsize=(10, 5))
    plt.plot(my_values, angleY_values, label='angleY vs my', color='blue')
    plt.xlabel('my (Y coordinate)')
    plt.ylabel('angleY (Servo Angle)')
    plt.title('Servo Angle Y vs Mouse Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
