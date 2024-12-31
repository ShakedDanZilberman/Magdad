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

STARTX = 55
STARTY = 45
deltaX = 30
deltaY = 10

angleX_rx_values = []  # To store [rx,angleX] values
angleY_ry_values = []


mouse_x, mouse_y = 0, 0


def click_event(event, x, y, flags, param):
    global mouse_x, mouse_y, click_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked coordinates: {relative_x}, {relative_y}")
        mouse_x, mouse_y = x, y
        click_flag = True  # Set flag to indicate a valid click



def bilerp(x0, y0, measurements):
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x = [item[0] for item in measurements]
    y = [item[1] for item in measurements]
    thetaX = [item[2] for item in measurements]
    thetaY = [item[3] for item in measurements]

    # get the four closest points to the black point in x,y space
    distance_squared = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2
    # use np because it's faster than list comprehension
    distances_squared = np.array([distance_squared(x0, y0, x[i], y[i]) for i in range(len(x))])
    closest_points = np.argsort(distances_squared)[:4]

    # Linearly interpolate the four closest points to get the value at the black point
    # Make sure to normalise it correctly

    angleX0 = 0
    angleY0 = 0
    for i in closest_points:
        angleX0 += thetaX[i] / distances_squared[i]
        angleY0 += thetaY[i] / distances_squared[i]

    angleX0 /= np.sum(1 / distances_squared[closest_points])
    angleY0 /= np.sum(1 / distances_squared[closest_points])

    return angleX0, angleY0

def get_data_from_measurements(measurements):
    x = [item[0] for item in measurements]
    y = [item[1] for item in measurements]
    thetaX = [item[2] for item in measurements]
    thetaY = [item[3] for item in measurements]
    return x, y, thetaX, thetaY

def offlineAnalysis(measurements):
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x, y, thetaX, thetaY = get_data_from_measurements(measurements)
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
    offlineAnalysis()




