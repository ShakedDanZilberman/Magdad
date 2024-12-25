import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pyfirmata import Arduino, util
from time import sleep


CAMERA_INDEX = 1
WINDOW_NAME = 'Camera Connection'


# Set up the Arduino board (replace 'COM8' with your Arduino's COM port)
board = Arduino('COM8')  # Adjust the COM port based on your system

# Define the pin for the servo (usually PWM pins)
servoV_pin = 5
servoH_pin = 3
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


A0 = -0.00000008
B0 = 0.000147
C0 = -0.17
D0 = 108

A1 = -0.000000323
B1 = 0.000246
C1 = -0.18
D1 = 71.5

# mx, my = 0, 0
# Main loop to control the servo

def bilerp(x0, y0):
    MEASUREMENTS = [(575, 416, 30.0, 20.0), (526, 413, 36.0, 20.0), (482, 410, 42.0, 20.0), (434, 407, 48.0, 20.0),
                    (387, 403, 54.0, 20.0), (337, 402, 60.0, 20.0), (287, 400, 66.0, 20.0), (241, 398,
                                                                                             72.0, 20.0),
                    (197, 397, 78.0, 20.0), (153, 396, 84.0, 20.0), (114, 396, 90.0, 20.0), (582, 394, 30.0, 24.0),
                    (530, 389, 36.0, 24.0), (485, 385, 42.0, 24.0), (438, 381, 48.0, 24.0),
                    (389, 378, 54.0, 24.0), (339, 375, 60.0, 24.0), (288, 372, 66.0, 24.0), (241, 371, 72.0, 24.0),
                    (196, 368, 78.0, 24.0), (151, 369, 84.0, 24.0), (111, 364, 90.0, 24.0), (591, 360,
                                                                                             30.0, 28.0),
                    (539, 353, 36.0, 28.0), (494, 349, 42.0, 28.0), (445, 345, 48.0, 28.0), (394, 342, 54.0, 28.0),
                    (343, 339, 60.0, 28.0), (290, 337, 66.0, 28.0), (243, 335, 72.0, 28.0),
                    (196, 333, 78.0, 28.0), (151, 333, 84.0, 28.0), (109, 331, 90.0, 28.0), (599, 326, 30.0, 32.0),
                    (547, 318, 36.0, 32.0), (500, 313, 42.0, 32.0), (451, 310, 48.0, 32.0), (400, 306,
                                                                                             54.0, 32.0),
                    (346, 303, 60.0, 32.0), (294, 299, 66.0, 32.0), (244, 298, 72.0, 32.0), (197, 295, 78.0, 32.0),
                    (150, 294, 84.0, 32.0), (108, 291, 90.0, 32.0), (605, 289, 30.0, 36.0),
                    (553, 282, 36.0, 36.0), (505, 279, 42.0, 36.0), (455, 275, 48.0, 36.0), (403, 272, 54.0, 36.0),
                    (350, 269, 60.0, 36.0), (297, 267, 66.0, 36.0), (247, 265, 72.0, 36.0), (200, 262,
                                                                                             78.0, 36.0),
                    (154, 261, 84.0, 36.0), (109, 259, 90.0, 36.0), (609, 256, 30.0, 40.0), (557, 251, 36.0, 40.0),
                    (509, 246, 42.0, 40.0), (457, 243, 48.0, 40.0), (405, 239, 54.0, 40.0),
                    (353, 237, 60.0, 40.0), (298, 234, 66.0, 40.0), (251, 232, 72.0, 40.0), (201, 231, 78.0, 40.0),
                    (156, 229, 84.0, 40.0), (111, 227, 90.0, 40.0), (611, 226, 30.0, 44.0), (559, 220,
                                                                                             36.0, 44.0),
                    (512, 216, 42.0, 44.0), (461, 212, 48.0, 44.0), (409, 209, 54.0, 44.0), (358, 206, 60.0, 44.0),
                    (301, 204, 66.0, 44.0), (253, 202, 72.0, 44.0), (204, 199, 78.0, 44.0),
                    (158, 199, 84.0, 44.0), (114, 195, 90.0, 44.0), (613, 196, 30.0, 48.0), (560, 191, 36.0, 48.0),
                    (511, 187, 42.0, 48.0), (462, 183, 48.0, 48.0), (412, 179, 54.0, 48.0), (358, 177,
                                                                                             60.0, 48.0),
                    (305, 174, 66.0, 48.0), (255, 173, 72.0, 48.0), (207, 171, 78.0, 48.0), (159, 170, 84.0, 48.0),
                    (117, 165, 90.0, 48.0), (613, 164, 30.0, 52.0), (561, 158, 36.0, 52.0),
                    (510, 154, 42.0, 52.0), (462, 150, 48.0, 52.0), (413, 147, 54.0, 52.0), (360, 144, 60.0, 52.0),
                    (309, 141, 66.0, 52.0), (259, 139, 72.0, 52.0), (212, 136, 78.0, 52.0), (165, 135,
                                                                                             84.0, 52.0),
                    (122, 132, 90.0, 52.0), (611, 137, 30.0, 56.0), (558, 131, 36.0, 56.0), (511, 128, 42.0, 56.0),
                    (463, 124, 48.0, 56.0), (414, 121, 54.0, 56.0), (364, 118, 60.0, 56.0),
                    (310, 116, 66.0, 56.0), (260, 114, 72.0, 56.0), (216, 111, 78.0, 56.0), (170, 110, 84.0, 56.0),
                    (125, 106, 90.0, 56.0), (607, 103, 30.0, 60.0), (557, 98, 36.0, 60.0), (510, 94, 42.0, 60.0),
                    (464, 91, 48.0, 60.0), (417, 87, 54.0, 60.0), (366, 85, 60.0, 60.0), (314, 83, 66.0, 60.0),
                    (268, 81, 72.0, 60.0), (220, 78, 78.0, 60.0), (176, 77, 84.0, 60.0), (134, 71, 90.0, 60.0)]

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


def angle_calc(coordinates):
    X = coordinates[0]  # rx
    Y = coordinates[1]  # ry

    # Coefficients for angleX polynomial
    A0 = 113.9773
    B0 = -0.0588
    C0 = 0.0001
    D0 = 0
    E0 = -0.1736
    F0 = 0.0001
    G0 = 0.0000
    H0 = 0.0001
    I0 = -0.0000
    J0 = 0

    # Coefficients for angleY polynomial
    A1 = 69.3912
    B1 = -0.1502
    C1 = 0.0001
    D1 = 0
    E1 = 0.0144
    F1 = 0.0
    G1 = 0.0000
    H1 = 0.0000
    I1 = 0.0000
    J1 = 0

    # Calculate angleX using the full polynomial expression
    angleX = (A0 + B0 * Y + C0 * Y ** 2 + D0 * Y ** 3 +
              E0 * X + F0 * X * Y + G0 * X * Y ** 2 +
              H0 * X ** 2 + I0 * X ** 2 * Y + J0 * X ** 3)

    # Calculate angleY using the full polynomial expression
    angleY = (A1 + B1 * Y + C1 * Y ** 2 + D1 * Y ** 3 +
              E1 * X + F1 * X * Y + G1 * X * Y ** 2 +
              H1 * X ** 2 + I1 * X ** 2 * Y + J1 * X ** 3)

    return angleX, angleY


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


# PID constants
Kp = -0.1
Ki = -0.0
Kd = 0.0

# Initialize previous values for PID
time_prev = time.time() / 100
integral = np.array([0, 0])
error_prev = np.array([0, 0])

def PID(target, curr, Kp=Kp, Ki=Ki, Kd=Kd):
    # target and curr are [x, y]
    global integral, time_prev, error_prev

    now = time.time() / 100
    error = np.array([target - curr]) 

    P = Kp * error
    integral = integral + Ki * error * (now - time_prev)
    D = Kd*(error - error_prev) / (now - time_prev) 
    delta = P + integral + D 
    error_prev = error
    time_prev = now
    # offset for the angles
    return delta


mx,my = 320, 240
mouse_x, mouse_y = 320, 240
flag = False


def click_event(event, x, y, flags, param):
    global mouse_x,mouse_y, mx, my
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x,mouse_y=x,y
        mx, my = x, y

def main():
    global mouse_x,mouse_y, flag
    #create camera and nonesense
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)
    # make sure there is an image to be read\sent
    ret_val, img = cam.read()
    cv2.setMouseCallback(WINDOW_NAME, click_event)
    if img.size > 0:
        cv2.imshow(WINDOW_NAME, img)
    angleX = 80
    angleY = 50
    # main loop
    while True:
        # read image
        ret_val, img = cam.read()
        global mx, my

        cv2.setMouseCallback(WINDOW_NAME, click_event)

        # display circles for laser and mouse
        (laser_x,laser_y) = find_red_point(img)
        cv2.circle(img,(laser_x,laser_y),7,(0,0,255),-1)
        cv2.circle(img,(mx,my),7,(255,0,0),-1)

        #magic numbers!!!
        # angleX = 180*(1/2-math.atan((mouse_x-laser_x)/340)/math.pi)
        # angleY = 180*(1/2-math.atan((mouse_y-laser_y)/340)/math.pi)
        
        
        # pid = PID(np.array([mouse_x,mouse_y]),np.array([laser_x,laser_y]))
        # angleX, angleY = pid[0,0], pid[0,1]


        cv2.circle(img,(mouse_x,mouse_y),7,(255,0,0),-1)

        # display image 
        cv2.imshow(WINDOW_NAME, img)
        
        angleX, angleY = angle_calc([mouse_x,mouse_y])
        if flag:
            pid = PID(np.array([mouse_x,mouse_y]),np.array([laser_x,laser_y]))
            print(pid)
            angleX += pid[0,0]
            angleY += pid[0,1]
            if angleX > 180: angleX = 180
            if angleY > 180: angleY = 180
            if angleX < 0: angleX = 0
            if angleY < 0: angleY = 0


        time.sleep(0.1)
    

        # Press Escape or close the window to exit
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        

        # angleX, angleY = angle_calc([mx-rx,my-ry])
        #magic numbers!!!
        print(mx,my)
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




