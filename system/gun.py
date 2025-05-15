import os
import sys
import serial
from time import sleep
import time
import subprocess
import re

from constants import COM


STEPS_IN_DEGREE = 2/1.8


class Gun:
    def __init__(self, gun_location, index: int, print_flag=False):
        """Initialize the Gun class, connect to the Arduino, and set the initial angle.
        The class communicates with the Arduino via a serial connection.
        The protocol:
        - The Gun class sends commands to the Arduino in the format "SHOOT\n" or "ROTATE:<angle>\n" (where <angle> is the number of steps).
        - The Gun class reads the Arduino's response line by line until it receives "Done".
        Sent commands are printed to the console with a ">>>" prefix.
        Received responses are printed with a "<<<" prefix.

        Args:
            print_flag (bool, optional): If True, print debug information. Defaults to False.
        """
        self.gun_index = index
        self.current_angle = 0
        self.gun_location = gun_location
        self.target_stack = []
        self.ser = self._connect_to_serial(COM)
        print("Connected to serial")
        time.sleep(2)  # Give Arduino time to reset; setup delay sleep for 2 seconds
        self.print_flag = print_flag

        if self.print_flag:
            print(f"Gun initialized and connected at {COM}.")

    def set_next_target(self, target):
        self.next_target = target

    def get_gun_location(self):
        """Get the location of the gun.

        Returns:
            tuple: The location of the gun as (x, y) coordinates.
        """
        return self.gun_location

    def _connect_to_serial(self, port):
        try:
            return serial.Serial(port, 9600, timeout=1)
        except serial.SerialException:
            # Try to auto-detect the port
            result = subprocess.run(
                ["mode"], capture_output=True, text=True, shell=True
            ).stdout
            device_pattern = r"Status for device (\w+):"
            devices = re.findall(device_pattern, result)
            if devices:
                return serial.Serial(devices[0], 9600, timeout=1)
            else:
                print("Arduino not connected or COM port is wrong")
                os.system("mode")
                sys.exit()

    def shoot(self):
        # print(f"Shooting!!! at {self.current_angle} degrees")
        self.ser.write(b"SHOOT\n")
        if self.print_flag:
            print(f">>> SHOOT")
        self._wait_for_done()

    def rotate(self, angle):
        """Rotate the gun to a specified angle.
        The angle is specified in degrees, and the gun will rotate to that angle from its current position.
        The angle is absolute, not relative.

        Args:
            angle (int): The angle to rotate to, in degrees.
        """
        print(f"Rotating to {angle} degrees")
        dθ = angle - self.current_angle
        steps = int(STEPS_IN_DEGREE * dθ)
        command = f"ROTATE:{steps}\n".encode()
        self.ser.write(command)
        if self.print_flag:
            print(f">>> {command}")
        self._wait_for_done()
        self.current_angle = angle 
        # print(f"Gun rotated to {self.current_angle} degrees")


    def is_free(self):
        if len(self.target_stack) <2:
            return True
        return False
    
    def get_angle(self):
        """Get the current angle of the gun.

        Returns:
            int: The current angle of the gun.
        """
        return self.current_angle

    def _wait_for_done(self):
        TIMEOUT = 10  # messages
        count = 0
        while True:
            response = self.ser.readline().decode().strip()
            if self.print_flag:
                print("<<<", response)
            count += 1
            if response == "Done":
                break
            if count > TIMEOUT:
                raise TimeoutError("Timeout waiting for Arduino response \"Done\" in Gun class.")

    def exit(self):
        pass

class DummyGun:
    # This class has exactly the same interface as Gun, but does nothing.
    def __init__(self):
        pass

    def shoot(self):
        print("DummyGun: Shooting")

    def rotate(self, angle):
        print(f"DummyGun: Rotating to {angle} degrees")
        pass

    def exit(self):
        pass


if __name__ == "__main__":
    gun = Gun((3,4), 0, print_flag=True)
    #angle_program = [0, 360, 0, 180, 0, 90, 0, -90, 0, 180, 0, 360, 0, -180, 0, 90, 0, -90, 0, 180, 0, 360, 0, -180, 0, 90, 0, -90, 0, 180, 0, 360, 0, -180, 0, 90, 0, -90, 0, 180, 0, 360]
    angle_program = [0,10,-15,25,-20]
    #angle_program *= 5
    while True:
        for angle in angle_program:
            gun.rotate(angle)
            sleep(0.5)
            gun.shoot()
            sleep(0.5)
