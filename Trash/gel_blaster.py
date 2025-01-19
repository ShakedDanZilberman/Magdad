import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pyfirmata import Arduino, util
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
board = Arduino('COM7')  # Adjust the COM port based on your system

# Define the pin for the servo (usually PWM pins)
gun_pin = 10
gun_servo_pin = 4
# Start an iterator thread to read analog inputs
it = util.Iterator(board)
it.start()
gun_servo = board.get_pin(f'd:{gun_servo_pin}:s')
def gun_shot():
    board.digital[gun_pin].write(1)
    sleep(0.2)
    board.digital[gun_pin].write(0)
def gun_move(deg):
    deg *= 240 / 180  # The servo thinks in terms of 0-180 degrees, but the servo can move 240 degrees
    gun_servo.write(deg)
def main():
    while True:
        gun_shot()
        sleep(5)
        gun_move(10)
        gun_shot()
        sleep(5)
        gun_move(180)

if __name__ == '__main__':
    main()