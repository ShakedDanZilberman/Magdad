import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import itertools
import time

ENTER = 13
ESC = 27

# Full MEASUREMENTS data (not truncated)
MEASUREMENTS = [
    (607, 255, 25.0, 35.0),
    (589, 243, 25.0, 40.0),
    (592, 208, 25.0, 45.0),
    (597, 168, 25.0, 50.0),
    (596, 135, 25.0, 55.0),
    (555, 249, 32.5, 35.0),
    (543, 237, 32.5, 40.0),
    (544, 203, 32.5, 45.0),
    (547, 164, 32.5, 50.0),
    (549, 130, 32.5, 55.0),
    (483, 241, 40.0, 35.0),
    (474, 225, 40.0, 40.0),
    (476, 194, 40.0, 45.0),
    (479, 157, 40.0, 50.0),
    (482, 122, 40.0, 55.0),
    (417, 233, 47.5, 35.0),
    (411, 221, 47.5, 40.0),
    (416, 186, 47.5, 45.0),
    (420, 149, 47.5, 50.0),
    (420, 117, 47.5, 55.0),
    (342, 225, 55.0, 35.0),
    (340, 210, 55.0, 40.0),
    (344, 180, 55.0, 45.0),
    (348, 143, 55.0, 50.0),
    (352, 103, 55.0, 55.0),
    (276, 217, 62.5, 35.0),
    (280, 205, 62.5, 40.0),
    (280, 174, 62.5, 45.0),
    (287, 137, 62.5, 50.0),
    (293, 104, 62.5, 55.0),
    (206, 211, 70.0, 35.0),
    (216, 196, 70.0, 40.0),
    (216, 171, 70.0, 45.0),
    (220, 133, 70.0, 50.0),
    (228, 99, 70.0, 55.0),
    (146, 205, 77.5, 35.0),
    (158, 199, 77.5, 40.0),
    (163, 169, 77.5, 45.0),
    (168, 132, 77.5, 50.0),
    (175, 97, 77.5, 55.0),
    (83, 202, 85.0, 35.0),
    (95, 189, 85.0, 40.0),
    (100, 161, 85.0, 45.0),
    (105, 121, 85.0, 50.0),
    (114, 88, 85.0, 55.0),
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

    STARTX = 55
    STARTY = 45
    deltaX = 30
    deltaY = 10
    NUM_ITERX = 8
    NUM_ITERY = 4

    angles = [
        (
            STARTX - deltaX + i * 2 * deltaX / NUM_ITERX,
            STARTY - deltaY + j * 2 * deltaY / NUM_ITERY,
        )
        for i, j in itertools.product(range(NUM_ITERX + 1), range(NUM_ITERY + 1))
    ]
    angles = iter(angles)
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
            for i, line in enumerate(text.split("\n")):
                cv2.putText(frame, line, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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
            if key == 8:
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


def bilerp(x0, y0, measurements):
    # Extract the x, y, thetaX, and thetaY values from MEASUREMENTS
    x = [item[0] for item in measurements]
    y = [item[1] for item in measurements]
    thetaX = [item[2] for item in measurements]
    thetaY = [item[3] for item in measurements]

    # get the four closest points to the black point in x,y space
    distance_squared = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2
    # use np because it's faster than list comprehension
    distances_squared = np.array(
        [distance_squared(x0, y0, x[i], y[i]) for i in range(len(x))]
    )
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
    angleX_eval = evaluate_polynomial(x_eval, y_eval, coeffsX, degree=3)
    angleY_eval = evaluate_polynomial(x_eval, y_eval, coeffsY, degree=3)

    # Visualization
    fig = plt.figure(figsize=(14, 10))

    # Plot angleX data
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(x, y, thetaX, color="red", label="angleX Data", s=100)
    ax1.plot_surface(
        x_eval, y_eval, angleX_eval, cmap="viridis", alpha=0.7, edgecolor="none"
    )
    ax1.set_title("3rd-Degree Polynomial Fit for angleX")
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
    ax2.set_title("3rd-Degree Polynomial Fit for angleY")
    ax2.set_xlabel("rx")
    ax2.set_ylabel("ry")
    ax2.set_zlabel("angleY")
    ax2.legend()

    plt.show()

    print("Polynomial for angleX:")
    print(print_polynomial(coeffsX, degree=3, var1="rx", var2="ry"))
    print(coeffsX)
    print("\nPolynomial for angleY:")
    print(print_polynomial(coeffsY, degree=3, var1="rx", var2="ry"))
    print(coeffsY)


if __name__ == "__main__":
    measure()
