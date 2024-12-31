from pyfirmata import Arduino, util
import cv2
import numpy as np

STARTX = 60
STARTY = 40
deltaX = 30
deltaY = 20
NUMITER = 10

class LaserPointer:
    """
    Class for controlling the laser pointer.
    """
    servoV_pin = 5
    servoH_pin = 3
    laser_pin = 8

    def __init__(self):
        """
        Initializes the LaserPointer.
        Steps of initialization:
        1. Connect to the Arduino board.
        2. Define the pins for the servos and the laser.
        3. Attach the servos to the board.
        4. Start an iterator thread to read analog inputs.
        """
        self.point = (0, 0)
        self.board = Arduino("COM8")

        self.board.digital[LaserPointer.laser_pin].write(1)
        # Attach the servo to the board
        self.servoV = self.board.get_pin(f"d:{LaserPointer.servoV_pin}:s")  # 's' means it's a servo
        self.servoH = self.board.get_pin(f"d:{LaserPointer.servoH_pin}:s")

        # Start an iterator thread to read analog inputs
        it = util.Iterator(self.board)
        it.start()


    def angle_from_coordinates(self, coordinates):
        """
        Calculate the angles for the servos from the given coordinates.
        Polynomial coefficients are determined experimentally.

        Args:
            coordinates (tuple): The coordinates of the point to move to

        Returns:
            tuple: The angles for the servos: (angleX, angleY)
        """
        X = coordinates[0]  # rx
        Y = coordinates[1]  # ry

        # Calculate angleX using the full polynomial expression
        angleX = 0

        # Calculate angleY using the full polynomial expression
        angleY = 0

        return angleX, angleY

    def move(self, point):
        """
        Move the laser pointer to the given point.

        Args:
            point (tuple): The point to move to
        """
        self.point = point
        angleX, angleY = self.angle_from_coordinates(point)
        # print(angleX, angleY)
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
