import os
import sys
import serial
from time import sleep
import time
import subprocess
import re

from constants import COM
import fit
import pid


class Gun:
    def __init__(self, print_flag=False):
        self.current_angle = 0

        self.ser = self._connect_to_serial(COM)
        time.sleep(2)  # Give Arduino time to reset; setup delay sleep for 2 seconds

        if print_flag:
            print(f"Gun initialized and connected at {COM}.")

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
        print("Shooting!!!")
        self.ser.write(b"SHOOT\n")
        print(f">>>SHOOT")
        self._wait_for_done()

    def rotate(self, angle):
        steps = angle - self.current_angle
        command = f"ROTATE:{steps}\n".encode()
        self.ser.write(command)
        print(f">>>{command}")
        self._wait_for_done()
        self.current_angle = angle

    def _wait_for_done(self):
        while True:
            response = self.ser.readline().decode().strip()
            print("<<<", response)
            if response == "Done":
                break

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
    gun = Gun(print_flag=True)
    gun.rotate(100)
    sleep(1)
    gun.shoot()
    sleep(1)
    gun.rotate(240)
    sleep(1)
    gun.shoot()
    sleep(1)
    gun.exit()
