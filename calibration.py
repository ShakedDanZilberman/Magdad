import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pyfirmata import Arduino, util
from time import sleep

CAMERA_INDEX = 1
WINDOW_NAME = 'Camera Connection'
MAX_CAMERAS = 10

# Set up the Arduino board (replace 'COM8' with your Arduino's COM port)
board = Arduino('COM7')  # Adjust the COM port based on your system

# Define the pin for the servo (usually PWM pins)
servoV_pin = 5
servoH_pin = 3  # Servo control pin (could be any PWM pin)
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
STARTX = 60
STARTY = 40
deltaX = 30
deltaY = 20

NUM_ITER = 10
A0 = 0
B0 = 0
C0 = 0.5
D0 = 90

A1 = 0
B1 = 0
C1 = 0.5
D1 = 90


angleX_rx_values = []  # To store [rx,angleX] values
angleY_ry_values = []

# Main loop to control the servo
def angle_calc(coordinates):
    X = coordinates[0]
    Y = coordinates[1]
    angleX = D0 + C0 * X + B0 * X ** 2 + A0 * X ** 3
    angleY = D1 + C1 * X + B1 * Y ** 2 + A1 * Y ** 3
    return angleX, angleY


mx, my = 0, 0


def click_event(event, x, y, flags, param):
    global mx, my
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked coordinates: {relative_x}, {relative_y}")
        mx, my = x, y


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


def main():
    servoH.write(STARTX)
    servoV.write(STARTY)
    sleep(1)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)

    ret_val, img = cam.read()
    sleep(1)
    global mx, my

    cv2.setMouseCallback(WINDOW_NAME, click_event)
    for j in range(NUM_ITER+1):
        for i in range(NUM_ITER+1):
            angleX = STARTX - deltaX + i*2*deltaX/NUM_ITER
            print(angleX)
            angleY = STARTY - deltaY + j*2*deltaY/NUM_ITER
            print(angleY)
            servoH.write(angleX)
            servoV.write(angleY)
            sleep(1)
            ret_val, img = cam.read()
            rx, ry = find_red_point(img)
            cv2.circle(img, (rx, ry), 10, (0, 0, 255), -1)
            angleX_rx_values.append([rx,angleX])
            angleY_ry_values.append([ry, angleY])
            sleep(0.5)
            # Press Escape or close the window to exit
            if cv2.waitKey(1) == 27:
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # angleX, angleY = angle_calc([mx-rx,my-ry])
            # magic numbers!!!
            cv2.imshow(WINDOW_NAME, img)
    cv2.destroyAllWindows()
    board.exit()
    rx_values = [item[0] for item in angleX_rx_values]
    angleX_values = [item[1] for item in angleX_rx_values]

    # Extract ry and angleY values from angleY_ry_values
    ry_values = [item[0] for item in angleY_ry_values]
    angleY_values = [item[1] for item in angleY_ry_values]

    # Compute 3rd degree polynomial fit for angleX vs rx
    fit_angleX = np.polyfit(rx_values, angleX_values, 3)  # 3rd degree polynomial fit
    fit_angleX_poly = np.poly1d(fit_angleX)

    # Compute 3rd degree polynomial fit for angleY vs ry
    fit_angleY = np.polyfit(ry_values, angleY_values, 3)  # 3rd degree polynomial fit
    fit_angleY_poly = np.poly1d(fit_angleY)

    # Generate x-values for smooth curve plotting
    rx_smooth = np.linspace(min(rx_values), max(rx_values), 500)
    ry_smooth = np.linspace(min(ry_values), max(ry_values), 500)
    # Print the fit parameters for angleX vs rx
    print("Fit Parameters for angleX vs rx (3rd-degree polynomial):")
    print(f"y = {fit_angleX[0]:.5e}x³ + {fit_angleX[1]:.5e}x² + {fit_angleX[2]:.5e}x + {fit_angleX[3]:.5e}")
    print(f"Coefficients: {fit_angleX}")

    # Print the fit parameters for angleY vs ry
    print("\nFit Parameters for angleY vs ry (3rd-degree polynomial):")
    print(f"y = {fit_angleY[0]:.5e}x³ + {fit_angleY[1]:.5e}x² + {fit_angleY[2]:.5e}x + {fit_angleY[3]:.5e}")
    print(f"Coefficients: {fit_angleY}")

    # Plot angleX vs rx with 3rd-degree polynomial fit
    plt.figure(figsize=(10, 5))
    plt.scatter(rx_values, angleX_values, color='green', label='Data points', alpha=0.8)
    plt.plot(rx_smooth, fit_angleX_poly(rx_smooth), color='red',
             label=f'3rd-degree fit: y = {fit_angleX[0]:.2e}x³ + {fit_angleX[1]:.2e}x² + {fit_angleX[2]:.2e}x + {fit_angleX[3]:.2e}')
    plt.xlabel('rx (Red Point X-coordinate)')
    plt.ylabel('angleX (Servo Angle X)')
    plt.title('Servo Angle X vs Red Point X-coordinate with 3rd Degree Polynomial Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot angleY vs ry with 3rd-degree polynomial fit
    plt.figure(figsize=(10, 5))
    plt.scatter(ry_values, angleY_values, color='blue', label='Data points', alpha=0.8)
    plt.plot(ry_smooth, fit_angleY_poly(ry_smooth), color='red',
             label=f'3rd-degree fit: y = {fit_angleY[0]:.2e}x³ + {fit_angleY[1]:.2e}x² + {fit_angleY[2]:.2e}x + {fit_angleY[3]:.2e}')
    plt.xlabel('ry (Red Point Y-coordinate)')
    plt.ylabel('angleY (Servo Angle Y)')
    plt.title('Servo Angle Y vs Red Point Y-coordinate with 3rd Degree Polynomial Fit')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()




