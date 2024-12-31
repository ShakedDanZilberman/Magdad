import time
import numpy as np

# PID constants
Kp = 0.1
Ki = 0.1
Kd = 0

class PID:
    def __init__(self):
        """
        Initialize PID controller

        Initializes the PID controller with the previous time, integral and error values set to 0
        """
        # Initialize previous values for PID
        self.time_prev = time.time() / 100
        self.integral = np.array([0, 0])
        self.error_prev = np.array([0, 0])

    def set_constants(new_Kp, new_Ki, new_Kd):
        """
        Set new PID constants

        Args:
            new_Kp (float): new Proportional constant
            new_Ki (float): new Integral constant
            new_Kd (float): new Derivative constant
        """
        global Kp, Ki, Kd
        Kp = new_Kp
        Ki = new_Ki
        Kd = new_Kd
        return

    def PID(target: tuple[float, float], curr: tuple[float, float], Kp=Kp, Ki=Ki, Kd=Kd):
        """
        PID controller for the system, returns the angle to turn to reach the target position

        Args:
            target (tuple[float, float]): target position to reach
            curr (tuple[float, float]): current position of the system
            Kp (float, optional): Proportional constant. Defaults to Kp.
            Ki (float, optional): Integral constant. Defaults to Ki.
            Kd (float, optional): Derivative constant. Defaults to Kd.

        Returns:
            float: angle to turn to reach the target position
        """
        # target and curr are (x, y)
        global integral, time_prev, error_prev

        now = time.time() / 100
        error = np.array([target - curr])

        P = Kp * error
        I = integral = integral + Ki * error * (now - time_prev)
        D = Kd * (error - error_prev) / (now - time_prev)

        error_prev = error
        time_prev = now
        return P + I + D
