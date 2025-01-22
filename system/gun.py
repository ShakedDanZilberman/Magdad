from pyfirmata import Arduino, util
import os
import sys

import serial
from time import sleep

from constants import COM


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
        self.servo.write(angle)

    def get_voltage(self):
        """
        Returns the voltage of the battery.

        Returns:
            float: The voltage of the battery.
        """
        return self.voltage_sensor.read()
    
    def shoot_target(self, target):
        # while True:
        #     # Move the laser pointer to the target
        #     if target is not None:
        #         thetaX, expected_volt = fit.bilerp(*center)
        #         # use PID
        #         motor_volt = gun.get_voltage()
        #         error = PID(expected_volt,motor_volt)  
        #         self.rotate(thetaX + error)
        #         time.sleep(0.1)
        #         self.shoot()
        #         print("Shooting", center)
        #         time.sleep(1)
        #         target_manager.clear()
        pass

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
