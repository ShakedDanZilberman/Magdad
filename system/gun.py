from pyfirmata import Arduino, util
import os
import sys

import serial
from time import sleep


class Gun:
    def __init__(self):
        """
        Initializes the Gun.
        Steps of initialization:
        1. Connect to the Arduino board.
        2. Define the pins for the gun and the servo.
        3. Attach the servo to the board.
        4. Start an iterator thread.
        """

        self.gun_pin = 10
        self.servo_pin = 4
        try:
            self.board = Arduino("COM7")
        except serial.serialutil.SerialException as e:
            print("Arduino not connected or COM port is wrong")
            # print the output of "mode" command in the CMD
            os.system("mode")
            sys.exit()
        it = util.Iterator(self.board)
        it.start()
        self.servo = self.board.get_pin(f'd:{self.servo_pin}:s')
        
    def shoot(self):
        """
        Shoots the gun, assuming there is a gel-blaster ball in the chamber.
        Contains a delay of 0.2 seconds.

        Returns:
            None
        """
        self.board.digital[self.gun_pin].write(1)
        sleep(0.2)
        self.board.digital[self.gun_pin].write(0)

    def rotate(self, angle):
        """
        Rotates the servo to the given angle.

        Args:
            angle (int): The angle to rotate to, in degrees, inside [0, 240].

        Returns:
            None
        """
        if not (0 <= angle <= 240):
            print(f"WARNING: The angle {angle} must be in the range [0, 240].")
        angle *= 180 / 240  # The servo thinks in terms of 0-180 degrees, but the servo can move 240 degrees
        self.servo.write(angle)

    def exit(self):
        pass