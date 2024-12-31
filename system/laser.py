from pyfirmata import Arduino, util
import cv2
import numpy as np
import serial
import fit
import sys
import os
# TODO - change import to class? maybe not? maybe it's okay

class LaserPointer:
    """
    Class for controlling the laser pointer.
    """
    servoV_pin = 5
    servoH_pin = 3
    laser_pin = 8
    sensorPin = 20


    def __init__(self):
        """
        Initializes the LaserPointer.
        Steps of initialization:
        1. Connect to the Arduino board.
        2. Define the pins for the servos and the laser.
        3. Attach the servos to the board.
        4. Start an iterator thread to read analog inputs.
        5. Create polynom for fitting
        """
        self.point = (0, 0)
        # you can check the correct port in the CMD with the command: mode
        try:
            self.board = Arduino("COM8")
        except serial.serialutil.SerialException as e:
            print("Arduino not connected or COM port is wrong")
            # print the output of "mode" command in the CMD
            os.system("mode")
            # sys.exit()

        self.board.digital[LaserPointer.laser_pin].write(1)
        # Attach the servo to the board
        self.servoV = self.board.get_pin(f"d:{LaserPointer.servoV_pin}:s")  # 's' means it's a servo
        self.servoH = self.board.get_pin(f"d:{LaserPointer.servoH_pin}:s")

        # Start an iterator thread to read analog inputs
        it = util.Iterator(self.board)
        it.start()

        # get polyonm
        # measure?
        self.coeffsX, self.coeffsY = fit.get_coeefs()


    def angle_from_coordinates(self, coordinates):
        """
        Calculate the angles for the servos from the given coordinates.
        Polynomial coefficients are determined experimentally.

        Args:
            coordinates (tuple): The coordinates of the point to move to

        Returns:
            tuple: The angles for the servos: (angleX, angleY)
        """
        # Calculate angleX using the full polynomial expression
        angleX = fit.evaluate_polynomial(coordinates[0], coordinates[1], self.coeffsX)

        # Calculate angleY using the full polynomial expression
        angleY = fit.evaluate_polynomial(coordinates[0], coordinates[1], self.coeffsY)

        return angleX, angleY

    def move(self, point):
        """
        Move the laser pointer to the given point.

        Args:
            point (tuple): The point to move to
        """
        self.point = point
        angleX, angleY = self.angle_from_coordinates(point)
        self.servoH.write(angleX)
        self.servoV.write(angleY)

    def turn_off(self):
        """
        Turn off the laser pointer.
        """
        self.board.digital[LaserPointer.laser_pin].write(0)

    def turn_on(self):
        """
        Turn on the laser pointer.
        """
        self.board.digital[LaserPointer.laser_pin].write(1)
        
    def exit(self):
        """
        Exit the Arduino board.
        """
        self.board.exit()

    def distance(self):
        """
        Get the distance from the sensor.
        """
        #Read the analog value from sensor
        sensorValue = self.board.get_pin(f"a:{LaserPointer.sensorPin}:i").read()

        #Convert the analog value to voltage
        voltage = sensorValue * (5.0 / 1023.0)

        #Convert the voltage to distance (cm)
        distance = 0.6301 * pow(voltage, -1.17)

        return distance