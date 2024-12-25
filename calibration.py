import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pyfirmata import Arduino, util
from time import sleep
from mpl_toolkits.mplot3d import Axes3D

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
STARTX = 55
STARTY = 45
deltaX = 30
deltaY = 10

NUM_ITERX = 8

NUM_ITERY = 3
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


mouse_x, mouse_y = 0, 0


def click_event(event, x, y, flags, param):
    global mouse_x, mouse_y, click_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked coordinates: {relative_x}, {relative_y}")
        mouse_x, mouse_y = x, y
        click_flag = True  # Set flag to indicate a valid click



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
    #TODO: make into a function 
    servoH.write(STARTX)
    servoV.write(STARTY)
    sleep(1)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)

    global mouse_x, mouse_y, click_flag

    # Initialize variables for clicked coordinates and flag
    mouse_x, mouse_y = 0, 0
    click_flag = False

    # Set mouse callback
    cv2.setMouseCallback(WINDOW_NAME, click_event)

    for j in range(NUM_ITERY + 1):
        for i in range(NUM_ITERX + 1):
            # Compute servo angles
            angleX = STARTX - deltaX + i * 2 * deltaX / NUM_ITERX
            angleY = STARTY - deltaY + j * 2 * deltaY / NUM_ITERY
            print(f"Servo angles: angleX={angleX}, angleY={angleY}")

            # Move the servos
            servoH.write(angleX)
            servoV.write(angleY)

            # Wait for the servos to stabilize
            sleep(1)

            # Capture the current frame from the camera
            ret_val, img = cam.read()
            if not ret_val:
                print("Failed to capture frame from camera.")
                continue

            # Reset click_flag and display the image until a click is detected
            click_flag = False
            while not click_flag:
                cv2.imshow(WINDOW_NAME, img)
                if cv2.waitKey(1) == 27:  # Break if 'ESC' is pressed
                    break

            # Mark the clicked point on the image
            cv2.circle(img, (mouse_x, mouse_y), 10, (0, 0, 255), -1)

            # Save the clicked coordinates with corresponding servo angles
            angleX_rx_values.append([mouse_x, angleX])
            angleY_ry_values.append([mouse_y, angleY])

            print(f"Saved click: mx={mouse_x}, my={mouse_y}, angleX={angleX}, angleY={angleY}")

            # Wait briefly before moving to the next position
            sleep(0.5)

            # Check if the window is closed
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    # Cleanup
    cv2.destroyAllWindows()
    board.exit()

if __name__ == '__main__':
    main()




