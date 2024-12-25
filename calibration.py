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

# Main loop to control the servo
def angle_calc(coordinates):
    X = coordinates[0]
    Y = coordinates[1]
    angleX = D0 + C0 * X + B0 * X ** 2 + A0 * X ** 3
    angleY = D1 + C1 * X + B1 * Y ** 2 + A1 * Y ** 3
    return angleX, angleY


mx, my = 0, 0


def click_event(event, x, y, flags, param):
    global mx, my, click_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked coordinates: {relative_x}, {relative_y}")
        mx, my = x, y
        click_flag = True  # Set flag to indicate a valid click


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
def polyfit2d(x, y, z, degree=3):
    """
    Fits a 2D polynomial of the given degree to the data (x, y, z).

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates.
        z: 1D array of dependent variable (z-values).
        degree: Degree of the polynomial (default is 3).

    Returns:
        coeffs: Coefficients of the 2D polynomial.
    """
    # Generate the design matrix for a 2D polynomial of the given degree
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))
    A = np.vstack(terms).T

    # Solve the least squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs
def evaluate2dpoly(coeffs, x, y, degree=3):
    """
    Evaluates a 2D polynomial with the given coefficients.

    Args:
        coeffs: Coefficients of the 2D polynomial.
        x: 2D array of x-coordinates.
        y: 2D array of y-coordinates.
        degree: Degree of the polynomial (default is 3).

    Returns:
        z: 2D array of predicted z-values.
    """
    z = np.zeros_like(x)
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z += coeffs[idx] * (x ** i) * (y ** j)
            idx += 1
    return z


def main():
    servoH.write(STARTX)
    servoV.write(STARTY)
    sleep(1)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)

    global mx, my, click_flag

    # Initialize variables for clicked coordinates and flag
    mx, my = 0, 0
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
            cv2.circle(img, (mx, my), 10, (0, 0, 255), -1)

            # Save the clicked coordinates with corresponding servo angles
            angleX_rx_values.append([mx, angleX])
            angleY_ry_values.append([my, angleY])

            print(f"Saved click: mx={mx}, my={my}, angleX={angleX}, angleY={angleY}")

            # Wait briefly before moving to the next position
            sleep(0.5)

            # Check if the window is closed
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    # Cleanup
    cv2.destroyAllWindows()
    board.exit()

    # Extract rx and angleX values from angleX_rx_values
    rx_values = [itemX[0] for itemX in angleX_rx_values]
    angleX_values = [itemX[1] for itemX in angleX_rx_values]
    ry_values = [itemY[0] for itemY in angleY_ry_values]
    angleY_values = [itemY[1] for itemY in angleY_ry_values]
    # rx_values = [item[0] for item in MEASUREMENTS]
    # ry_values = [item[1] for item in MEASUREMENTS]
    # angleX_values = [item[2] for item in MEASUREMENTS]
    # angleY_values = [item[3] for item in MEASUREMENTS]


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
    with open("fit_parameters.txt", "w") as file:
        file.write("Fit Parameters for angleX vs rx (3rd-degree polynomial):\n")
        file.write(
            f"y = {fit_angleX[0]:.5e}x³ + {fit_angleX[1]:.5e}x² + {fit_angleX[2]:.5e}x + {fit_angleX[3]:.5e}\n\n")

        file.write("Fit Parameters for angleY vs ry (3rd-degree polynomial):\n")
        file.write(
            f"y = {fit_angleY[0]:.5e}x³ + {fit_angleY[1]:.5e}x² + {fit_angleY[2]:.5e}x + {fit_angleY[3]:.5e}\n\n")

        file.write("Fit Parameters for angleX vs ry (3rd-degree polynomial):\n")
        file.write(
            f"y = {fit_angleX_ry[0]:.5e}x³ + {fit_angleX_ry[1]:.5e}x² + {fit_angleX_ry[2]:.5e}x + {fit_angleX_ry[3]:.5e}\n\n")

        file.write("Fit Parameters for angleY vs rx (3rd-degree polynomial):\n")
        file.write(
            f"y = {fit_angleY_rx[0]:.5e}x³ + {fit_angleY_rx[1]:.5e}x² + {fit_angleY_rx[2]:.5e}x + {fit_angleY_rx[3]:.5e}\n\n")

    print("Fit parameters saved to 'fit_parameters.txt'.")

    # Print the fit parameters
       # Fit a 3rd-degree 2D polynomial for angleY vs rx, ry
    coeffs_angleY = polyfit2d(np.array(rx_values), np.array(ry_values), np.array(angleY_values), degree=3)

    # Fit a 3rd-degree 2D polynomial for angleX vs rx, ry
    coeffs_angleX = polyfit2d(np.array(rx_values), np.array(ry_values), np.array(angleX_values), degree=3)

    # Generate a grid for visualization
    rx_grid, ry_grid = np.meshgrid(np.linspace(min(rx_values), max(rx_values), 100),
                                   np.linspace(min(ry_values), max(ry_values), 100))

    # Evaluate the fitted 2D polynomials
    angleY_pred = evaluate2dpoly(coeffs_angleY, rx_grid, ry_grid, degree=3)
    angleX_pred = evaluate2dpoly(coeffs_angleX, rx_grid, ry_grid, degree=3)

    # Plot 3D surface for angleY vs rx, ry
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rx_values, ry_values, angleY_values, c='blue', label='Data points')
    ax.plot_surface(rx_grid, ry_grid, angleY_pred, cmap='viridis', alpha=0.6)
    ax.set_xlabel('rx (Red Point X-coordinate)')
    ax.set_ylabel('ry (Red Point Y-coordinate)')
    ax.set_zlabel('angleY (Servo Angle Y)')
    ax.set_title('3D Polynomial Fit: angleY vs rx, ry')
    ax.legend()
    plt.show()

    # Plot 3D surface for angleX vs rx, ry
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rx_values, ry_values, angleX_values, c='green', label='Data points')
    ax.plot_surface(rx_grid, ry_grid, angleX_pred, cmap='plasma', alpha=0.6)
    ax.set_xlabel('rx (Red Point X-coordinate)')
    ax.set_ylabel('ry (Red Point Y-coordinate)')
    ax.set_zlabel('angleX (Servo Angle X)')
    ax.set_title('3D Polynomial Fit: angleX vs rx, ry')
    ax.legend()
    plt.show()
    # Save the coefficients to a text file
    with open("2d_polynomial_fit_coeffs.txt", "w") as file:
        file.write("3rd-Degree 2D Polynomial Fit Coefficients:\n\n")

        file.write("angleY = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 + c6*x^3 + c7*x^2*y + c8*x*y^2 + c9*y^3\n")
        file.write("Coefficients for angleY vs rx, ry:\n")
        for i, coeff in enumerate(coeffs_angleY):
            file.write(f"c{i}: {coeff:.5e}\n")
        file.write("\n")

        file.write("angleX = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 + c6*x^3 + c7*x^2*y + c8*x*y^2 + c9*y^3\n")
        file.write("Coefficients for angleX vs rx, ry:\n")
        for i, coeff in enumerate(coeffs_angleX):
            file.write(f"c{i}: {coeff:.5e}\n")
if __name__ == '__main__':
    main()




