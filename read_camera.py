import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from pyfirmata import Arduino, util
from time import sleep


CAMERA_INDEX = 1
WINDOW_NAME = 'Camera Connection'
MAX_CAMERAS = 10

# Set up the Arduino board (replace 'COM8' with your Arduino's COM port)
board = Arduino('COM7')  # Adjust the COM port based on your system
# you can check on windows by the "mode" command in the CMD

# Define the pin for the servo (usually PWM pins)
servoV_pin = 4
servoH_pin = 3# Servo control pin (could be any PWM pin)
laser_pin = 8
board.digital[laser_pin].write(1)
# Attach the servo to the board
servoV = board.get_pin(f'd:{servoV_pin}:s')  # 's' means it's a servo
servoH = board.get_pin(f'd:{servoH_pin}:s')
# Start an iterator thread to read analog inputs
it = util.Iterator(board)
it.start()


# PID shit
A0 = 0
B0 = 0
C0 = 0
D0 = 90

A1 = 0
B1 = 0
C1 = 0
D1 = 90
# Main loop to control the servo
def angle_calc(coordinates):
    X = coordinates[0]
    Y = coordinates[1]
    angleX = D0 + C0*X +B0*X**2 + A0*X**3
    angleY = D1 + C1*X +B1*Y**2 + A1*Y**3
    if angleX>180:angleX=180
    if angleY>180:angleY=180
    if angleX<0:angleX=0
    if angleY<0:angleY=0
    return angleX, angleY

# PID constants (tune these based on your system)
Kp = .01  # Proportional gain
Ki = 0.00  # Integral gain
Kd = 0.00  # Derivative gain

# Initialize previous values for PID
prev_errorX = 0
prev_errorY = 0
integralX = 0
integralY = 0
dt = 0.01  # Time step (seconds)


def calculate_PID_coefficients(errorX, errorY):
    global prev_errorX, prev_errorY, integralX, integralY

    # Proportional term
    P_X = Kp * errorX
    P_Y = Kp * errorY

    # Integral term
    integralX += errorX * dt
    integralY += errorY * dt
    I_X = Ki * integralX
    I_Y = Ki * integralY

    # Derivative term
    derivativeX = (errorX - prev_errorX) / dt
    derivativeY = (errorY - prev_errorY) / dt
    D_X = Kd * derivativeX
    D_Y = Kd * derivativeY

    # Update previous errors
    prev_errorX = errorX
    prev_errorY = errorY

    # Calculate coefficients
    global A0, B0, C0, D0, A1, B1, C1, D1
    C0 = P_X + I_X + D_X
    B0 = 0  # Modify based on specific requirements
    A0 = 0  # Modify based on specific requirements
    C1 = P_Y + I_Y + D_Y
    B1 = 0  # Modify based on specific requirements
    A1 = 0  # Modify based on specific requirements

        
# straight from chatGPT
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


mouse_x,mouse_y = 0, 0

def click_event(event, x, y, flags, param):
    global mouse_x,mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Clicked coordinates: {relative_x}, {relative_y}")
        mouse_x,mouse_y=x,y     

def main():
    global mouse_x,mouse_y
    #create camera and nonesense
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow(WINDOW_NAME)
    # make sure there is an image to be read\sent
    ret_val, img = cam.read()
    cv2.setMouseCallback(WINDOW_NAME, click_event)

    # main loop
    while True:
        # read image
        ret_val, img = cam.read()
        # display circles for laser and mouse
        (red_point_x,red_point_y) = find_red_point(img)
        cv2.circle(img,(red_point_x,red_point_y),10,(0,0,255),-1)
        cv2.circle(img,(mouse_x,mouse_y),10,(255,0,0),-1)

        #magic numbers!!!
        angleX = 180*(1/2-math.atan((mouse_x-red_point_x)/340)/math.pi)
        angleY = 180*(1/2-math.atan((mouse_y-red_point_y)/340)/math.pi)
        calculate_PID_coefficients(mouse_x-red_point_x,mouse_y-red_point_y)
        # angleX, angleY = angle_calc([mx-rx,my-ry])

        servoH.write(angleX)
        sleep(0.1)
        servoV.write(angleY)
        sleep(0.1)

        # display image 
        cv2.imshow(WINDOW_NAME, img)


        
        # Press Escape or close the window to exit
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    board.exit()


if __name__ == '__main__':
    main()




