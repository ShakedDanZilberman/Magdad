from pyfirmata import Arduino, util
import cv2
import numpy as np

# Coefficients for angleX polynomial
AX = 70
BX = -0.05
CX = 0.0001
DX = 0
EX = -0.1736
FX = 0.0001
GX = 0.0000
HX = 0.0001
IX = -0.0000
JX = 0

# Coefficients for angleY polynomial
AY = 55.3912
BY = -0.1502
CY = 0.0001
DY = 0
EY = 0.0144
FY = 0.0
GY = 0.0000
HY = 0.0000
IY = 0.0000
JY = 0

STARTX = 60
STARTY = 40
deltaX = 30
deltaY = 20
NUMITER = 10

class LaserPointer:
    """
    Class for controlling the laser pointer.
    """
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

        # Define the pins for the servos and the laser
        servoV_pin = 5
        servoH_pin = 3
        laser_pin = 8
        self.board.digital[laser_pin].write(1)
        # Attach the servo to the board
        self.servoV = self.board.get_pin(f"d:{servoV_pin}:s")  # 's' means it's a servo
        self.servoH = self.board.get_pin(f"d:{servoH_pin}:s")

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
        angleX = (
            AX
            + BX * Y
            + CX * Y**2
            + DX * Y**3
            + EX * X
            + FX * X * Y
            + GX * X * Y**2
            + HX * X**2
            + IX * X**2 * Y
            + JX * X**3
        )

        # Calculate angleY using the full polynomial expression
        angleY = (
            AY
            + BY * Y
            + CY * Y**2
            + DY * Y**3
            + EY * X
            + FY * X * Y
            + GY * X * Y**2
            + HY * X**2
            + IY * X**2 * Y
            + JY * X**3
        )

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
