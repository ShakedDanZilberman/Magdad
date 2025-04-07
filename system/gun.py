from pyfirmata import Arduino, util
import os
import sys

import serial
from time import sleep
import time

from constants import COM

import fit
import pid

class Gun:
    def __init__(self, print_flag=False):
        """
        Initializes the Gun.
        Steps of initialization:
        1. Connect to the Arduino board.
        2. Define the pins for the gun and the servo.
        3. Attach the servo to the board.
        4. Start an iterator thread.
        """
        self.voltage_motor_pin = 4
        self.gun_pin = 4
        self.servo_pin = 9
        self.sleep_duration = 0.2
        try:
            self.board = Arduino(COM)

        except serial.serialutil.SerialException as e:
            import subprocess
            import re
            result = subprocess.run(["mode"], capture_output=True, text=True, shell=True).stdout
            device_pattern = r"Status for device (\w+):"
            devices = re.findall(device_pattern, result)
            if devices is not []:
                self.board = Arduino(devices[0])
            else:
                print("Arduino not connected or COM port is wrong")
                # print the output of "mode" command in the CMD
                os.system("mode")
                sys.exit()
        it = util.Iterator(self.board)
        it.start()
        self.servo = self.board.get_pin(f"d:{self.servo_pin}:s")
        self.voltage_sensor = self.board.analog[self.voltage_motor_pin]
        self.voltage_sensor.enable_reporting()
        if print_flag:
            print("Gun initialised and connected.")


    def shoot(self):
        """
        Shoots the gun, assuming there is a gel-blaster ball in the chamber.
        Contains a delay of 0.2 seconds.

        Returns:
            None
        """
        print("Shooting!!!")
        self.board.digital[self.gun_pin].write(1)
        sleep(self.sleep_duration)
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
        angle = int(angle)
        angle *= 180 / 240  # The servo thinks in terms of 0-180 degrees, but the servo can move 240 degrees
        # print("Rotating to angle", angle)
        self.servo.write(angle)

    def get_voltage(self):
        """
        Returns the voltage of the battery.

        Returns:
            float: The voltage of the battery.
        """
        return self.voltage_sensor.read()
    
    def aim_and_fire_target(self, target):

        P_ERROR = -75
        I_ERROR = -28
        D_ERROR = 0
        fix = 0
        
        fixer = pid.PID(P_ERROR, I_ERROR, D_ERROR)

        thetaX, expected_volt = fit.bilerp(*target)
        self.rotate(thetaX)

        start_time = time.time()
        print("Start PID")
        # Run the loop for 1.5 second
        while time.time() - start_time < 1.5:
            self.rotate(thetaX + fix)
            print("PIDing to", thetaX + fix)
            motor_volt = self.get_voltage()
            if motor_volt is not None:
                fix = fixer.PID(expected_volt,motor_volt) 
        sleep(0.1)
        print("Finished PID")
        self.shoot()
        return

    def exit(self):
        pass


class DummyGun:
    # This class has exactly the same interface as Gun, but does nothing.
    def __init__(self):
        pass

    def shoot(self):
        print("DummyGun: Shooting")

    def rotate(self, angle):
        pass

    def exit(self):
        pass
