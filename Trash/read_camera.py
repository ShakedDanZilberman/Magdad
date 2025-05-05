import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from pyfirmata import Arduino, util
from time import sleep


CAMERA_INDEX_0 = 1
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
    # exists in calibratin.py
    return
def find_red_point():
    # exsits in lasering.py
    return
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
    cam = cv2.VideoCapture(CAMERA_INDEX_0)
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




