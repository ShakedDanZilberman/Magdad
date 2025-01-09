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
deltaX = 5
deltaY = 5
ENDX = 105
ENDY = 85

# Full MEASUREMENTS data (not truncated)
MEASUREMENTS = [
    (611, 234, 35, 42),
    (611, 173, 35, 50),
    (609, 116, 35, 58),
    (602, 59, 35, 66),
    (591, 2, 35, 74),
    (602, 332, 50, 18),
    (553, 281, 50, 26),
    (524, 251, 50, 34),
    (500, 228, 50, 42),
    (504, 164, 50, 50),
    (501, 106, 50, 58),
    (498, 50, 50, 66),
    (493, 1, 50, 74),
    (431, 368, 65, 10),
    (413, 329, 65, 18),
    (393, 276, 65, 26),
    (381, 243, 65, 34),
    (371, 224, 65, 42),
    (374, 161, 65, 50),
    (375, 103, 65, 58),
    (377, 48, 65, 66),
    (375, 1, 65, 74),
    (200, 407, 80, 10),
    (220, 325, 80, 18),
    (236, 270, 80, 26),
    (245, 239, 80, 34),
    (253, 214, 80, 42),
    (258, 159, 80, 50),
    (260, 101, 80, 58),
    (267, 46, 80, 66),
    (267, 0, 80, 74),
    (13, 383, 95, 10),
    (56, 306, 95, 18),
    (89, 265, 95, 26),
    (117, 236, 95, 34),
    (140, 213, 95, 42),
    (146, 158, 95, 50),
    (152, 103, 95, 58),
    (161, 45, 95, 66),
    (167, 0, 95, 74),
    
]


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


def measure():
    mouseX, mouseY = 0, 0
    from laser import LaserPointer
    from cameraIO import Camera
    from main import CAMERA_INDEX

    laser_pointer = LaserPointer()
    camera = Camera(CAMERA_INDEX)
    title = "Camera Feed"

    def on_mouse(event, x, y, flags, param):
        nonlocal mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    measurements = []
    WAIT_FOR_KEY = 1  # milliseconds

    rangeX = range(STARTX, ENDX + 1, deltaX)
    rangeY = range(STARTY, ENDY + 1, deltaY)

    angles = iter(itertools.product(rangeX, rangeY))
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
            # convert the frame to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.circle(frame, (mouseX, mouseY), 7, (255, 0, 0), -1)
            if laserX is not None and laserY is not None:
                cv2.circle(frame, (laserX, laserY), 7, (0, 0, 255), -1)
            # add text at the top left corner
            text = "Enter chooses RED\nSpace chooses MOUSE\nBackspace skips angle"
            textcolor = (255, 255, 255)
            for i, line in enumerate(text.split("\n")):
                cv2.putText(frame, line, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1)
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
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x = np.array([item[0] for item in MEASUREMENTS])
    y = np.array([item[1] for item in MEASUREMENTS])
    thetaX = np.array([item[2] for item in MEASUREMENTS])
    thetaY = np.array([item[3] for item in MEASUREMENTS])

    # get the four closest points to the black point in x,y space
    distance_squared = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2
    # use np because it's faster than list comprehension
    distances_squared = [distance_squared(x0, y0, x[i], y[i]) for i in range(len(x))]
    distance_squared = np.array(distances_squared)

    # get the indices of the four closest points
    closest_points = np.argsort(distances_squared)[:4]

    # Linearly interpolate the four closest points to get the value at the black point
    # Make sure to normalise it correctly

    angleX0 = 0
    angleY0 = 0
    assert len(closest_points) == 4
    for i in closest_points:
        if distances_squared[i] == 0:
            return thetaX[i], thetaY[i]
        distance = np.sqrt(distances_squared[i])
        weight = 1 / distance
        angleX0 += weight * thetaX[i]
        angleY0 += weight * thetaY[i]

    normalizer = sum([1 / np.sqrt(distances_squared[i]) for i in closest_points])

    angleX0 /= normalizer
    angleY0 /= normalizer

    return angleX0, angleY0


def fit_3d_polynomial(x, y, z, degree=3):
    """
    Fits a 3rd-degree 2D polynomial f(x, y) = z using least squares regression.
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
    x = [item[0] for item in measurements]
    y = [item[1] for item in measurements]
    thetaX = [item[2] for item in measurements]
    thetaY = [item[3] for item in measurements]
    return x, y, thetaX, thetaY


def get_coeefs(measurements=MEASUREMENTS):
    x, y, thetaX, thetaY = get_data_from_measurements(measurements)
    # Fit the 3D polynomial to the data
    coeffsX = fit_3d_polynomial(x, y, thetaX, degree=3)
    coeffsY = fit_3d_polynomial(x, y, thetaY, degree=3)
    return coeffsX, coeffsY


def evaluate(x, y, coeffsX, coeffsY):
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


def main():

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


if __name__ == "__main__":
    measure()
