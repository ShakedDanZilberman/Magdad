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
board = Arduino('COM8')  # Adjust the COM port based on your system

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

MEASUREMENTS = [(575, 416, 30.0, 20.0), (526, 413, 36.0, 20.0), (482, 410, 42.0, 20.0), (434, 407, 48.0, 20.0), (387, 403, 54.0, 20.0), (337, 402, 60.0, 20.0), (287, 400, 66.0, 20.0), (241, 398,
72.0, 20.0), (197, 397, 78.0, 20.0), (153, 396, 84.0, 20.0), (114, 396, 90.0, 20.0), (582, 394, 30.0, 24.0), (530, 389, 36.0, 24.0), (485, 385, 42.0, 24.0), (438, 381, 48.0, 24.0),
 (389, 378, 54.0, 24.0), (339, 375, 60.0, 24.0), (288, 372, 66.0, 24.0), (241, 371, 72.0, 24.0), (196, 368, 78.0, 24.0), (151, 369, 84.0, 24.0), (111, 364, 90.0, 24.0), (591, 360,
30.0, 28.0), (539, 353, 36.0, 28.0), (494, 349, 42.0, 28.0), (445, 345, 48.0, 28.0), (394, 342, 54.0, 28.0), (343, 339, 60.0, 28.0), (290, 337, 66.0, 28.0), (243, 335, 72.0, 28.0),
 (196, 333, 78.0, 28.0), (151, 333, 84.0, 28.0), (109, 331, 90.0, 28.0), (599, 326, 30.0, 32.0), (547, 318, 36.0, 32.0), (500, 313, 42.0, 32.0), (451, 310, 48.0, 32.0), (400, 306,
54.0, 32.0), (346, 303, 60.0, 32.0), (294, 299, 66.0, 32.0), (244, 298, 72.0, 32.0), (197, 295, 78.0, 32.0), (150, 294, 84.0, 32.0), (108, 291, 90.0, 32.0), (605, 289, 30.0, 36.0),
 (553, 282, 36.0, 36.0), (505, 279, 42.0, 36.0), (455, 275, 48.0, 36.0), (403, 272, 54.0, 36.0), (350, 269, 60.0, 36.0), (297, 267, 66.0, 36.0), (247, 265, 72.0, 36.0), (200, 262,
78.0, 36.0), (154, 261, 84.0, 36.0), (109, 259, 90.0, 36.0), (609, 256, 30.0, 40.0), (557, 251, 36.0, 40.0), (509, 246, 42.0, 40.0), (457, 243, 48.0, 40.0), (405, 239, 54.0, 40.0),
 (353, 237, 60.0, 40.0), (298, 234, 66.0, 40.0), (251, 232, 72.0, 40.0), (201, 231, 78.0, 40.0), (156, 229, 84.0, 40.0), (111, 227, 90.0, 40.0), (611, 226, 30.0, 44.0), (559, 220,
36.0, 44.0), (512, 216, 42.0, 44.0), (461, 212, 48.0, 44.0), (409, 209, 54.0, 44.0), (358, 206, 60.0, 44.0), (301, 204, 66.0, 44.0), (253, 202, 72.0, 44.0), (204, 199, 78.0, 44.0),
 (158, 199, 84.0, 44.0), (114, 195, 90.0, 44.0), (613, 196, 30.0, 48.0), (560, 191, 36.0, 48.0), (511, 187, 42.0, 48.0), (462, 183, 48.0, 48.0), (412, 179, 54.0, 48.0), (358, 177,
60.0, 48.0), (305, 174, 66.0, 48.0), (255, 173, 72.0, 48.0), (207, 171, 78.0, 48.0), (159, 170, 84.0, 48.0), (117, 165, 90.0, 48.0), (613, 164, 30.0, 52.0), (561, 158, 36.0, 52.0),
 (510, 154, 42.0, 52.0), (462, 150, 48.0, 52.0), (413, 147, 54.0, 52.0), (360, 144, 60.0, 52.0), (309, 141, 66.0, 52.0), (259, 139, 72.0, 52.0), (212, 136, 78.0, 52.0), (165, 135,
84.0, 52.0), (122, 132, 90.0, 52.0), (611, 137, 30.0, 56.0), (558, 131, 36.0, 56.0), (511, 128, 42.0, 56.0), (463, 124, 48.0, 56.0), (414, 121, 54.0, 56.0), (364, 118, 60.0, 56.0),
 (310, 116, 66.0, 56.0), (260, 114, 72.0, 56.0), (216, 111, 78.0, 56.0), (170, 110, 84.0, 56.0), (125, 106, 90.0, 56.0), (607, 103, 30.0, 60.0), (557, 98, 36.0, 60.0), (510, 94, 42.0, 60.0),
(464, 91, 48.0, 60.0), (417, 87, 54.0, 60.0), (366, 85, 60.0, 60.0), (314, 83, 66.0, 60.0), (268, 81, 72.0, 60.0), (220, 78, 78.0, 60.0), (176, 77, 84.0, 60.0), (134, 71, 90.0, 60.0)]


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

def bilerp(x0, y0):
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x = [item[0] for item in MEASUREMENTS]
    y = [item[1] for item in MEASUREMENTS]
    thetaX = [item[2] for item in MEASUREMENTS]
    thetaY = [item[3] for item in MEASUREMENTS]

    # get the four closest points to the black point in x,y space
    distance_squared = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2
    # use np because it's faster than list comprehension
    distances_squared = np.array([distance_squared(x0, y0, x[i], y[i]) for i in range(len(x))])
    closest_points = np.argsort(distances_squared)[:4]

    # Cubic interpolate the four closest points to get the value at the black point
    # Make sure to normalise it correctly
    
    angleX0 = 0
    angleY0 = 0
    for i in closest_points:
        angleX0 += thetaX[i] / distances_squared[i]
        angleY0 += thetaY[i] / distances_squared[i]

    angleX0 /= np.sum(1 / distances_squared[closest_points])
    angleY0 /= np.sum(1 / distances_squared[closest_points])

    return angleX0, angleY0

def offlineAnalysis():
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x = [item[0] for item in MEASUREMENTS]
    y = [item[1] for item in MEASUREMENTS]
    thetaX = [item[2] for item in MEASUREMENTS]
    thetaY = [item[3] for item in MEASUREMENTS]
    xspace = np.linspace(min(x), max(x), 100)
    yspace = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(xspace, yspace)
    thetaXspace = np.zeros((100, 100))
    thetaYspace = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            thetaXspace[i, j], thetaYspace[i, j] = bilerp(X[i, j], Y[i, j])
    

    # Display a heatmap with x and y as the spatial coordinates, and value thetaXspace, with pcolormesh
    fig, (ax1, ax2) = plt.subplots(2)
    pcm = ax1.pcolormesh(X, Y, thetaXspace, cmap='coolwarm')
    fig.colorbar(pcm, ax=ax1)
    ax1.set_xlabel('rx')
    ax1.set_ylabel('ry')
    ax1.set_title('Heatmap of thetaX')

    # Display a heatmap with rx and ry as the spatial coordinate, and value thetaY
    pcm = ax2.pcolormesh(X, Y, thetaYspace, cmap='YlGnBu')
    fig.colorbar(pcm, ax=ax2)

    ax2.set_xlabel('rx')
    ax2.set_ylabel('ry')
    ax2.set_title('Heatmap of thetaY')
    
    plt.show()


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
    if False:
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
                angleX_rx_values.append([rx, angleX])
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
    # Extract rx and angleX values from angleX_rx_values
    # rx_values = [item[0] for item in angleX_rx_values]
    # angleX_values = [item[1] for item in angleX_rx_values]
    rx_values = [item[0] for item in MEASUREMENTS]
    ry_values = [item[1] for item in MEASUREMENTS]
    angleX_values = [item[2] for item in MEASUREMENTS]
    angleY_values = [item[3] for item in MEASUREMENTS]


    # Extract ry and angleY values from angleY_ry_values
    # ry_values = [item[0] for item in angleY_ry_values]
    # angleY_values = [item[1] for item in angleY_ry_values]

    # print(list(zip(rx_values, ry_values, angleX_values, angleY_values)))

    # Compute 3rd degree polynomial fit for angleX vs rx
    fit_angleX = np.polyfit(rx_values, angleX_values, 3)  # 3rd degree polynomial fit
    fit_angleX_poly = np.poly1d(fit_angleX)

    # Compute 3rd degree polynomial fit for angleY vs ry
    fit_angleY = np.polyfit(ry_values, angleY_values, 3)  # 3rd degree polynomial fit
    fit_angleY_poly = np.poly1d(fit_angleY)

    # Compute 3rd degree polynomial fit for angleX vs ry
    fit_angleX_ry = np.polyfit(ry_values, angleX_values, 3)  # angleX vs ry
    fit_angleX_ry_poly = np.poly1d(fit_angleX_ry)

    # Compute 3rd degree polynomial fit for angleY vs rx
    fit_angleY_rx = np.polyfit(rx_values, angleY_values, 3)  # angleY vs rx
    fit_angleY_rx_poly = np.poly1d(fit_angleY_rx)

    # Generate x-values for smooth curve plotting
    rx_smooth = np.linspace(min(rx_values), max(rx_values), 500)
    ry_smooth = np.linspace(min(ry_values), max(ry_values), 500)

    # Print the fit parameters
    print("Fit Parameters for angleX vs rx (3rd-degree polynomial):")
    print(f"y = {fit_angleX[0]:.5e}x³ + {fit_angleX[1]:.5e}x² + {fit_angleX[2]:.5e}x + {fit_angleX[3]:.5e}")

    print("\nFit Parameters for angleY vs ry (3rd-degree polynomial):")
    print(f"y = {fit_angleY[0]:.5e}x³ + {fit_angleY[1]:.5e}x² + {fit_angleY[2]:.5e}x + {fit_angleY[3]:.5e}")

    print("\nFit Parameters for angleX vs ry (3rd-degree polynomial):")
    print(f"y = {fit_angleX_ry[0]:.5e}x³ + {fit_angleX_ry[1]:.5e}x² + {fit_angleX_ry[2]:.5e}x + {fit_angleX_ry[3]:.5e}")

    print("\nFit Parameters for angleY vs rx (3rd-degree polynomial):")
    print(f"y = {fit_angleY_rx[0]:.5e}x³ + {fit_angleY_rx[1]:.5e}x² + {fit_angleY_rx[2]:.5e}x + {fit_angleY_rx[3]:.5e}")

    # Plot angleX vs rx
    plt.figure(figsize=(10, 5))
    plt.scatter(rx_values, angleX_values, color='green', label='Data points', alpha=0.8)
    plt.plot(rx_smooth, fit_angleX_poly(rx_smooth), color='red', label='3rd-degree polynomial fit')
    plt.xlabel('rx (Red Point X-coordinate)')
    plt.ylabel('angleX (Servo Angle X)')
    plt.title('Servo Angle X vs Red Point X-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot angleY vs ry
    plt.figure(figsize=(10, 5))
    plt.scatter(ry_values, angleY_values, color='blue', label='Data points', alpha=0.8)
    plt.plot(ry_smooth, fit_angleY_poly(ry_smooth), color='red', label='3rd-degree polynomial fit')
    plt.xlabel('ry (Red Point Y-coordinate)')
    plt.ylabel('angleY (Servo Angle Y)')
    plt.title('Servo Angle Y vs Red Point Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot angleX vs ry
    plt.figure(figsize=(10, 5))
    plt.scatter(ry_values, angleX_values, color='purple', label='Data points', alpha=0.8)
    plt.plot(ry_smooth, fit_angleX_ry_poly(ry_smooth), color='red', label='3rd-degree polynomial fit')
    plt.xlabel('ry (Red Point Y-coordinate)')
    plt.ylabel('angleX (Servo Angle X)')
    plt.title('Servo Angle X vs Red Point Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot angleY vs rx
    plt.figure(figsize=(10, 5))
    plt.scatter(rx_values, angleY_values, color='orange', label='Data points', alpha=0.8)
    plt.plot(rx_smooth, fit_angleY_rx_poly(rx_smooth), color='red', label='3rd-degree polynomial fit')
    plt.xlabel('rx (Red Point X-coordinate)')
    plt.ylabel('angleY (Servo Angle Y)')
    plt.title('Servo Angle Y vs Red Point X-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()




