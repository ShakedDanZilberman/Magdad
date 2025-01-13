import time
import numpy as np


class PID:
    """
    PID controller for general use.
    Currently redundant, but can be used for future implementations.
    """
    def __init__(self, Kp=0.1, Ki=0.1, Kd=0):
        """
        Initialize PID controller

        Initializes the PID controller with the previous time, integral and error values set to 0
        """
        # PID constants
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Initialize previous values for PID
        self.time_prev = time.time() / 100
        self.integral = np.array([0, 0])
        self.error_prev = np.array([0, 0])

    def set_constants(self, new_Kp, new_Ki, new_Kd):
        """
        Set new PID constants

        Args:
            new_Kp (float): new Proportional constant
            new_Ki (float): new Integral constant
            new_Kd (float): new Derivative constant
        """
        self.Kp = new_Kp
        self.Ki = new_Ki
        self.Kd = new_Kd
        return

    def PID(self, target: tuple[float, float], curr: tuple[float, float]):
        """
        PID controller for the system, returns the angle to turn to reach the target position

        Args:
            target (tuple[float, float]): target position to reach
            curr (tuple[float, float]): current position of the system

        Returns:
            float: angle to turn to reach the target position
        """
        # target and curr are (x, y)
        now = time.time() / 100
        error = np.array([target - curr])

        P = self.Kp * error
        I = self.integral + self.Ki * error * (now - self.time_prev)
        D = self.Kd * (error - self.error_prev) / (now - self.time_prev)

        self.error_prev = error
        self.time_prev = now
        return P + I + D
