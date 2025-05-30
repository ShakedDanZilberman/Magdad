import os
import sys
import serial
from serial.tools import list_ports as coms
from time import sleep
import time
import subprocess
import re


class Gun:
    def __init__(self, gun_location, index: int, COM: str, print_flag=False):
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
        self.COM = COM
        self.ser = self._connect_to_serial(self.COM)
        print("Connected to serial")
        time.sleep(2)  # Give Arduino time to reset; setup delay sleep for 2 seconds
        self.print_flag = print_flag

        if self.print_flag:
            print(f"Gun initialized and connected at {self.COM}.")

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
            ports = coms.comports()
            for port in ports:
                print(port.device, end=": ")
                print(port.description)
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
        time.sleep(0.05)
        self.ser.write(b"SHOOT\n")
        time.sleep(0.05)
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
        # steps = int(STEPS_IN_DEGREE * dθ)
        # command = f"ROTATE:{steps}\n".encode()
        command = f"ANGLE:{dθ}\n".encode()
        time.sleep(0.5)  # Give Arduino time to process the command
        # print("in gun rotate: sleep 0.5 secs")
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
                raise TimeoutError("Timeout waiting for Arduino response \"Done\" in Gun class.\n Close Arduino IDE, disconnect and reconnect the cable and try again.\n Good Luck!")

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

def simple_test(gun):
    #angle_program = [0, 360, 0, 180, 0, 90, 0, -90, 0, 180, 0, 360, 0, -180, 0, 90, 0, -90, 0, 180, 0, 360, 0, -180, 0, 90, 0, -90, 0, 180, 0, 360, 0, -180, 0, 90, 0, -90, 0, 180, 0, 360]
    # # angle_program = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    angle_program = [30, 0, -30, 0, 60, 0, -60, 0, 90, 0, -90]
    # angle_program = [0]
    #angle_program *= 5
    while True:
        for angle in angle_program:
            gun.rotate(angle)
            sleep(0.5)
            gun.shoot()
            sleep(3.5)

if __name__ == "__main__":   
    white_gun = Gun(gun_location=(195.5,100.0),
                    index=0,
                    COM="COM14", 
                    print_flag=False)
    
    black_gun = Gun(gun_location=(100.0,95.0),
                    index=1,
                    COM="COM18", 
                    print_flag=False)
    
    gun_info = [((195.5,100.0), 0, "COM14", False), ((100.0,95.0), 1, "COM18", False)]
    guns = [white_gun, black_gun]
    # guns = []

    from Brain import Brain
    # guns = []
    # cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    from constants import *
    import threading
    cam_info = [(CAMERA_INDEX_0, CAMERA_LOCATION_0, homography_matrices[0]), 
                (CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1]), 
                (CAMERA_INDEX_2, CAMERA_LOCATION_2, homography_matrices[2])]  # (cam_index, CAMERA_LOCATION_0, homography_matrix)
    # cam_info = [(CAMERA_INDEX_1, CAMERA_LOCATION_1, homography_matrices[1])]
    try:
        brain = Brain(guns, cam_info)
        # brain.game_loop_independent()
        brain.game_loop_display()
    except KeyboardInterrupt:
        for thread in threading.enumerate():
            print(f"Thread {thread.name} is alive: {thread.is_alive()}")
        print("Exiting...")
        exit(0)
