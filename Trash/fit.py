import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import itertools
import time

ENTER = 13
ESC = 27
BACKSPACE = 8

STARTX = 20
STARTY = 10
deltaX = 2
deltaY = 5
ENDX = 105
ENDY = 85

# Full MEASUREMENTS data (not truncated)

MEASUREMENTS = [(612, 203, 46, 0.1535), (574, 201, 48, 0.1603), (551, 201, 50, 0.1642), (516, 201, 52, 0.172), 
                (490, 201, 54, 0.175), (446, 198, 56, 0.1808), (424, 198, 58, 0.1857), (380, 198, 60, 0.1926),
                  (363, 197, 62, 0.1965), (322, 197, 64, 0.2023), (304, 196, 66, 0.2053), (255, 198, 68, 0.2131),
                    (237, 199, 70, 0.217), (202, 201, 72, 0.2229), (180, 200, 74, 0.2297), (139, 204, 76, 0.2366), 
                    (124, 205, 78, 0.2424), (84, 205, 80, 0.2473), (67, 208, 82, 0.2502), (37, 210, 84, 0.2581), (24, 208, 86, 0.261)]
def find_red_point(frame):
    """
    Finds the (x, y) coordinates of the single red point in the image.

    Args:
        frame: The input image (BGR format).

    Returns:
        A tuple (x, y) representing the center of the red point if found, or None if no red point is detected.
    """
    # Convert the frame to HSV color space
    # Define the lower and upper bounds for red in grayscale
    lower_red = 120  # Lower range for red in grayscale
    upper_red = 255  # Upper range for red in grayscale

    # Create mask for red
    mask = cv2.inRange(frame, lower_red, upper_red)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return (0, 0)  # No red point found

    # Assume the largest contour is the red point (adjust as needed)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the center of the red point
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None  # Avoid division by zero

    cX = int(M["m10"] / M["m00"])  # x-coordinate
    cY = int(M["m01"] / M["m00"])  # y-coordinate

    return cX, cY


def measure_for_lidar():
    """
    Measure the angles for the laser pointer at each point in the grid.

    The user can click on the red point or the mouse cursor to add a measurement.
    Press Enter to add the red point to the measurements.
    Press Space to add the mouse cursor to the measurements.
    Press Backspace to skip the current angle.

    Press Esc to exit the program.

    After all measurements are taken, the program will print the measurements and exit.
    *TAKE THE PRINTED MEASUREMENTS AND COPY THEM INTO THE MEASUREMENTS LIST ABOVE*
    """
    mouseX, mouseY = 0, 0
    from laser import LaserPointer
    from cameraIO import Camera
    from constants import CAMERA_INDEX_0

    laser_pointer = LaserPointer()
    camera = Camera(CAMERA_INDEX_0)
    title = "Camera Feed"

    def on_mouse(event, x, y, flags, param):
        """
        Mouse callback function. Updates the global mouseX, mouseY variables to the position of the click.
        """
        nonlocal mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    measurements = []
    WAIT_FOR_KEY = 1  # milliseconds

    # The ranges of angles to measure
    rangeAngleX = range(STARTX, ENDX + 1, deltaX)
    rangeAngleY = range(STARTY, ENDY + 1, deltaY)

    angles = iter(itertools.product(rangeAngleX, rangeAngleY))
    nextAngleFlag = True

    try:
        while True:
            # Get the feed from the camera
            if nextAngleFlag:
                angleX, angleY = next(angles)
                print("Next angle: ", angleX, angleY)
                laser_pointer.move_raw(angleX, angleY)
                laser_pointer.turn_on()
                time.sleep(0.3)
                nextAngleFlag = False

            frame = camera.read()
            laserX, laserY = find_red_point(frame)
            # insert motor pin
            # convert the frame to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.circle(frame, (mouseX, mouseY), 7, (255, 0, 0), -1)
            if laserX is not None and laserY is not None:
                cv2.circle(frame, (laserX, laserY), 7, (0, 0, 255), -1)
            # add text at the top left corner
            text = "Enter chooses RED\nSpace chooses MOUSE\nBackspace skips angle"
            textcolor = (255, 255, 255)
            for i, line in enumerate(text.split("\n")):
                cv2.putText(
                    frame,
                    line,
                    (10, 20 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    textcolor,
                    1,
                )
            cv2.imshow(title, frame)

            # if user presses Enter then add the red point to the measurements
            key = cv2.waitKey(WAIT_FOR_KEY) & 0xFF
            if key == ENTER:
                measurements.append((laserX, laserY, angleX, angleY))
                nextAngleFlag = True
                print(f"Added measurement: ({laserX}, {laserY}, {angleX}, {angleY})")
            # If the user presses the spacebar, add the mouse point to the measurements
            if key == ord(" "):
                measurements.append((mouseX, mouseY, angleX, angleY))
                nextAngleFlag = True
                print(f"Added measurement: ({mouseX}, {mouseY}, {angleX}, {angleY})")
            # If the user presses backspace, skip the current angle
            if key == BACKSPACE:
                nextAngleFlag = True
                print("Skipped angle")
            # Esc key to exit
            if key == ESC:
                break
    except KeyboardInterrupt:
        print(measurements)
        laser_pointer.exit()
        raise
    except StopIteration:
        laser_pointer.exit()
        print("All angles measured:")

    print(measurements)
    cv2.destroyAllWindows()
    laser_pointer.exit()

def bilerp(x0, y0):
    """
    Perform bilinear interpolation to estimate the angleX and V_motor values at the given point (x0, y0).
    Calculates the weighted average of the two closest points to the given point.

    Args:
        x0 (float): The x-coordinate of the point.
        y0 (float): The y-coordinate of the point. Ignored; kept for backwards compatibility.

    Returns:
        tuple: The estimated angleX and V_motor values at the point (x0, y0).
    """
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x = np.array([item[0] for item in MEASUREMENTS])
    # y = np.array([item[1] for item in MEASUREMENTS])
    thetaX = np.array([item[2] for item in MEASUREMENTS])
    # thetaY = np.array([item[3] for item in MEASUREMENTS])
    V_motor = np.array([item[3] for item in MEASUREMENTS])

    if len(x) == 0:
        return 0, 0
    if len(x[x < x0]) == 0:
        left_pointX = x[0]
    else:
        left_pointX = np.max(x[x < x0])
    if len(x[x >= x0]) == 0:
        right_pointX = x[-1]
    else: 
        right_pointX = np.min(x[x >= x0])
    
    print("Left:", left_pointX)
    print("Right:", right_pointX)
    print("x0:", x0)
    
    left_point = np.where(x == left_pointX)[0]
    right_point = np.where(x == right_pointX)[0]

    thetaX_left = thetaX[left_point]
    thetaX_right = thetaX[right_point]
    V_motor_left = V_motor[left_point]
    V_motor_right = V_motor[right_point]

    distance_to_left = np.abs(x0 - x[left_point])
    distance_to_right = np.abs(x[right_point] - x0)

    # bilerp according to the distances
    angleX0 = (thetaX_left * distance_to_right + thetaX_right * distance_to_left) / (
        distance_to_left + distance_to_right
    )

    motor_voltage0 = (V_motor_left * distance_to_right + V_motor_right * distance_to_left) / (
        distance_to_left + distance_to_right
    )
    
    if angleX0 < thetaX_left:
        angleX0 = thetaX_left
    if angleX0 > thetaX_right:
        angleX0 = thetaX_right

    return angleX0, motor_voltage0


def fit_3d_polynomial(x, y, z, degree=3):
    """
    Fits a 3rd-degree 2D polynomial f(x, y) = z using least squares regression.
    Returns the coefficients of the polynomial.
    """

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("x, y, and z must have the same length")

    # Generate polynomial terms up to the specified degree
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x**i) * (y**j))

    # Combine terms into design matrix A
    A = np.vstack(terms).T

    # Solve the least squares problem to find the coefficients
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


def print_polynomial(coeffs, degree=3, var1="x", var2="y"):
    """
    Prints the polynomial equation given the coefficients.
    """
    terms = []
    term_idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            term = f"{coeffs[term_idx]:.4f}*{var1}^{i}*{var2}^{j}"
            terms.append(term)
            term_idx += 1
    polynomial = " + ".join(terms)
    return polynomial


def evaluate_polynomial(x, y, coeffs, degree=3):
    """
    Evaluates the fitted 2D polynomial at points (x, y).

    """
    x = np.array(x)
    y = np.array(y)
    z = np.zeros_like(x, dtype=float)
    # TODO: degree is dictated by the coeeffs
    term_idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z += coeffs[term_idx] * (x**i) * (y**j)
            term_idx += 1
    return z


def get_data_from_measurements(measurements):
    """
    Extracts the x, y, thetaX, and thetaY values from the measurements
    and returns them as separate lists.
    """
    x = [item[0] for item in measurements]
    y = [item[1] for item in measurements]
    thetaX = [item[2] for item in measurements]
    thetaY = [item[3] for item in measurements]
    return x, y, thetaX, thetaY


def get_coeefs(measurements=MEASUREMENTS):
    """
    Fits a 3rd-degree 2D polynomial to the measurements and returns the coefficients.
    """
    x, y, thetaX, thetaY = get_data_from_measurements(measurements)
    # Fit the 3D polynomial to the data
    coeffsX = fit_3d_polynomial(x, y, thetaX, degree=3)
    coeffsY = fit_3d_polynomial(x, y, thetaY, degree=3)
    return coeffsX, coeffsY


def evaluate(x, y, coeffsX, coeffsY):
    """
    Evaluates the 3rd-degree 2D polynomial at the given point (x, y).
    Returns the estimated angleX and angleY values.

    This is the most important function in this script,
    because it is used to estimate the angleX and angleY values
    at any point in the image space (x, y).

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        coeffsX (np.ndarray): The coefficients of the 3rd-degree polynomial for angleX.
        coeffsY (np.ndarray): The coefficients of the 3rd-degree polynomial for angleY.

    Returns:
        tuple: The estimated angleX and angleY values at the point (x, y).
    """
    angleXbilerp, angleYbilerp = bilerp(x, y)
    angleXpoly = evaluate_polynomial(x, y, coeffsX, degree=3)
    angleYpoly = evaluate_polynomial(x, y, coeffsY, degree=3)
    image_center = np.array([320, 240])
    position = np.array([x, y])
    distance_from_center = np.linalg.norm(position - image_center)

    effective_radius = 300
    interpolation_range = 50

    if distance_from_center < effective_radius:
        return angleXbilerp, angleYbilerp
    elif distance_from_center > effective_radius + interpolation_range:
        return angleXpoly, angleYpoly
    else:
        # interpolate between the two
        weight = (distance_from_center - effective_radius) / interpolation_range
        angleX = (1 - weight) * angleXbilerp + weight * angleXpoly
        angleY = (1 - weight) * angleYbilerp + weight * angleYpoly
        return angleX, angleY


def show_graphs():
    """
    Show the 3D polynomial fit for angleX and angleY.

    This function is used to visualize the 3D polynomial fit for angleX and angleY.
    It displays two 3D plots: one for angleX and one for angleY.

    It also displays the difference between the measurements and the polynomial fit,
    showing how well the polynomial fits the data.
    """
    # only needed to display the polynom
    measurements = MEASUREMENTS
    coeffsX, coeffsY = get_coeefs(measurements)
    x, y, thetaX, thetaY = get_data_from_measurements(measurements)
    # Create a meshgrid for evaluating the polynomial surface
    x_eval, y_eval = np.meshgrid(
        np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50)
    )

    # Evaluate the polynomial on the meshgrid
    # evaluate knows to handle only numbers, not arrays, so we need to iterate over the meshgrid
    # we want to store the results in 2 2D arrays, so we can plot them
    angleX_eval = np.zeros((50, 50))
    angleY_eval = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            angleX_eval[i, j], angleY_eval[i, j] = evaluate(
                x_eval[i, j], y_eval[i, j], coeffsX, coeffsY
            )

    # Visualization
    fig = plt.figure(figsize=(14, 10))

    # Plot angleX data
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(x, y, thetaX, color="red", label="angleX Data", s=100)
    ax1.plot_surface(
        x_eval, y_eval, angleX_eval, cmap="viridis", alpha=0.7, edgecolor="none"
    )
    ax1.set_title("3rd-Degree Hybrid Fit for angleX")
    ax1.set_xlabel("rx")
    ax1.set_ylabel("ry")
    ax1.set_zlabel("angleX")
    ax1.legend()

    # Plot angleY data
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(x, y, thetaY, color="blue", label="angleY Data", s=100)
    ax2.plot_surface(
        x_eval, y_eval, angleY_eval, cmap="viridis", alpha=0.7, edgecolor="none"
    )
    ax2.set_title("3rd-Degree Hybrid Fit for angleY")
    ax2.set_xlabel("rx")
    ax2.set_ylabel("ry")
    ax2.set_zlabel("angleY")
    ax2.legend()

    # Plot the difference between the measurements and the polynomial fit
    fig2 = plt.figure(figsize=(14, 10))
    ax3 = fig2.add_subplot(121, projection="3d")
    angleX_diff = evaluate_polynomial(x, y, coeffsX, degree=3) - thetaX
    ax3.scatter(x, y, angleX_diff, color="red", label="angleX Difference", s=100)
    ax3.set_title("Difference between angleX Data and Fit")
    ax3.set_xlabel("rx")
    ax3.set_ylabel("ry")
    ax3.set_zlabel("angleX Difference")
    ax3.legend()

    ax4 = fig2.add_subplot(122, projection="3d")
    angleY_diff = evaluate_polynomial(x, y, coeffsY, degree=3) - thetaY
    ax4.scatter(x, y, angleY_diff, color="blue", label="angleY Difference", s=100)
    ax4.set_title("Difference between angleY Data and Fit")
    ax4.set_xlabel("rx")
    ax4.set_ylabel("ry")
    ax4.set_zlabel("angleY Difference")
    ax4.legend()

    plt.show()

    print("Polynomial for angleX:")
    print(print_polynomial(coeffsX, degree=3, var1="rx", var2="ry"))
    print(coeffsX)
    print("\nPolynomial for angleY:")
    print(print_polynomial(coeffsY, degree=3, var1="rx", var2="ry"))
    print(coeffsY)

    # show a graph of x vs thetaX and x vs thetaY
    plt.figure(figsize=(14, 10))
    plt.subplot(121)
    plt.scatter(x, thetaX, color="red", label="Angle Data")
    plt.plot(x, evaluate_polynomial(x, y, coeffsX, degree=3), color="blue", label="Angle Fit")
    plt.xlabel("x")
    plt.ylabel("Angle")
    plt.title("Angle vs x")
    plt.legend()

    plt.subplot(122)
    plt.scatter(x, thetaY, color="red", label="Voltage Data")
    plt.plot(x, evaluate_polynomial(x, y, coeffsY, degree=3), color="blue", label="Voltage Fit")
    plt.xlabel("x")
    plt.ylabel("Voltage")
    plt.title("Voltage vs X")
    plt.legend()

    plt.show()


def display_grid(img=None, display=True):
    """
    Display the grid of points on the screen.
    Connects the points with lines.
    This is basically a grid in angleX, angleY space,
    cast onto the image (x, y) space.

    Args:
        img (np.ndarray, optional): The image to display the grid on. Defaults to None.
        display (bool, optional): Whether to display the image. Defaults to True.

    Returns:
        np.ndarray: The image with the grid drawn on it.
    """
    if img is None:
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # color it white
        img.fill(255)
    # add points to the image from the measurements
    overlay = img.copy()
    black = (0, 0, 0)
    radius = 4
    for x, y, angleX, angleY in MEASUREMENTS:
        cv2.circle(overlay, (x, y), radius, black, -1)
    # connect the points with lines
    # first, convert MEASUREMETS into a 2D array based on the angleX, angleY values
    
    # sort the measurements by angleX, angleY
    MEASUREMENTS.sort(key=lambda x: (x[2], x[3]))

    # We'd like to connect the points in a grid-like fashion

    # First, we need to find the unique angleX and angleY values
    angleX_values = set([item[2] for item in MEASUREMENTS])
    angleY_values = set([item[3] for item in MEASUREMENTS])

    # Now, we can iterate over the angleX and angleY values
    # and connect the points in a grid-like fashion

    # Connect the points in the x-direction
    for angleX in angleX_values:
        points = [item for item in MEASUREMENTS if item[2] == angleX]
        points.sort(key=lambda x: x[3])
        for i in range(len(points) - 1):
            x1, y1, _, _ = points[i]
            x2, y2, _, _ = points[i + 1]
            cv2.line(overlay, (x1, y1), (x2, y2), black, 2)

    # Connect the points in the y-direction
    for angleY in angleY_values:
        points = [item for item in MEASUREMENTS if item[3] == angleY]
        points.sort(key=lambda x: x[2])
        for i in range(len(points) - 1):
            x1, y1, _, _ = points[i]
            x2, y2, _, _ = points[i + 1]
            cv2.line(overlay, (x1, y1), (x2, y2), black, 2)

    # Apply the overlay with 50% opacity
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if display:
        # Display the image
        cv2.imshow("Grid", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return img



def measure_for_gun():
    """
    Measure the angles for the gel blaster at each point in the grid.

    The user can click on the red point or the mouse cursor to add a measurement.
    Press Enter to add the red point to the measurements.
    Press Space to add the mouse cursor to the measurements.
    Press Backspace to skip the current angle.

    Press Esc to exit the program.

    After all measurements are taken, the program will print the measurements and exit.
    *TAKE THE PRINTED MEASUREMENTS AND COPY THEM INTO THE MEASUREMENTS LIST ABOVE*
    """
    mouseX, mouseY = 0, 0
    from gun import Gun
    from cameraIO import Camera
    from system.main import CAMERA_INDEX_0

    gun = Gun()
    camera = Camera(CAMERA_INDEX_0)
    title = "Camera Feed"

    def on_mouse(event, x, y, flags, param):
        """
        Mouse callback function. Updates the global mouseX, mouseY variables to the position of the click.
        """
        nonlocal mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    measurements = []
    WAIT_FOR_KEY = 1  # milliseconds

    # The ranges of angles to measure
    rangeAngleX = range(STARTX, ENDX + 1, deltaX)

    angles = iter(rangeAngleX)
    nextAngleFlag = True

    try:
        while True:
            # Get the feed from the camera
            if nextAngleFlag:
                angleX = next(angles)
                print("Next angle: ", angleX)
                motor_value = gun.rotate(angleX)
                time.sleep(0.3)
                nextAngleFlag = False

            frame = camera.read()
            laserX, laserY = find_red_point(frame)
            # convert the frame to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.circle(frame, (mouseX, mouseY), 7, (255, 0, 0), -1)
            if laserX is not None and laserY is not None:
                cv2.circle(frame, (laserX, laserY), 7, (0, 0, 255), -1)
            # add text at the top left corner
            text = "Enter chooses RED\nSpace chooses MOUSE\nBackspace skips angle"
            textcolor = (255, 255, 255)
            for i, line in enumerate(text.split("\n")):
                cv2.putText(
                    frame,
                    line,
                    (10, 20 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    textcolor,
                    1,
                )
            cv2.imshow(title, frame)

            Voltage = gun.get_voltage()

            # if user presses Enter then add the red point to the measurements
            key = cv2.waitKey(WAIT_FOR_KEY) & 0xFF
            if key == ENTER:
                measurements.append((laserX, laserY, angleX,Voltage))
                nextAngleFlag = True
                print(f"Added measurement: ({laserX}, {laserY}, {angleX},{Voltage})")
            # If the user presses the spacebar, add the mouse point to the measurements
            if key == ord(" "):
                measurements.append((mouseX, mouseY, angleX,Voltage))
                nextAngleFlag = True
                print(f"Added measurement: ({mouseX}, {mouseY}, {angleX},{Voltage})")
            # If the user presses backspace, skip the current angle
            if key == BACKSPACE:
                nextAngleFlag = True
                print("Skipped angle")
            # Esc key to exit
            if key == ESC:
                break
    except KeyboardInterrupt:
        print(measurements)
        gun.exit()
        raise
    except StopIteration:
        gun.exit()
        print("All angles measured for gun:")

    print(measurements)
    cv2.destroyAllWindows()
    gun.exit()


if __name__ == "__main__":
    #measure_for_lidar()  # Uncomment this line to measure the angles.
    measure_for_gun()
    #show_graphs()
    # display_grid()
