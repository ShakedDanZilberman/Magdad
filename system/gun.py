from pyfirmata import Arduino, util
import os
import sys
import numpy as np
import serial
from time import sleep
import time

from constants import COM, SLEEP_DURATION, VOLTAGE_MOTOR_PIN, GUN_PIN, SERVO_PIN

import fit
import pid

PRECISION = 16.0 # should result in precision of one degree 

class Gun:
    def __init__(self, print_flag=False):
        """
        Initializes the Gun.
        Steps of initialization:
        1. Connect to the Arduino board.
        2. Define the pins for the gun and the servo in the constants file.
        3. Attach the servo to the board.
        4. Start an iterator thread.
        """
        self.gun_angle = 0
        try:
            self.board = Arduino(COM)

        except serial.serialutil.SerialException as e:
            print("Arduino not connected or COM port is wrong")
            # print the output of "mode" command in the CMD
            os.system("mode")
            sys.exit()
        it = util.Iterator(self.board)
        it.start()
        self.servo = self.board.get_pin(f"d:{SERVO_PIN}:s")
        self.voltage_sensor = self.board.analog[VOLTAGE_MOTOR_PIN]
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
        self.board.digital[GUN_PIN].write(1)
        sleep(SLEEP_DURATION)
        self.board.digital[GUN_PIN].write(0)

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
    
    def aim_and_fire_target(self, target):

        P_ERROR = -75
        I_ERROR = -30
        D_ERROR = 0
        fix = 0
        
        fixer = pid.PID(P_ERROR, I_ERROR, D_ERROR)

        thetaX, expected_volt = fit.bilerp(*target)
        self.rotate(thetaX)

        start_time = time.time()
        # Run the loop for 1 second
        # TODO - change time to global var
        while time.time() - start_time < 2:
            self.rotate(thetaX + fix)
            motor_volt = self.get_voltage()
            if motor_volt is not None:
                fix = fixer.PID(expected_volt,motor_volt) 
        sleep(0.5)
        self.shoot()
        return

    def aim_and_fire_target_2(self, target):

        P_ERROR = -75
        I_ERROR = -30
        D_ERROR = 0
        fix = 0
        
        fixer = pid.PID(P_ERROR, I_ERROR, D_ERROR)

        thetaX, expected_volt = fit.bilerp(*target)
        self.rotate(thetaX)

        start_time = time.time()
        # Run the loop for 1 second
        # TODO - change time to global var
        # TODO: get rid of excess delay - in PID end condition and time.sleep. There might be some necessary delay
        motor_volt = self.get_voltage()
        if motor_volt is None:
            motor_volt = 0
        while time.time() - start_time < 1 and np.abs(expected_volt - motor_volt) > PRECISION:
            self.rotate(thetaX + fix)
            motor_volt_temp = self.get_voltage()
            if motor_volt_temp is not None:
                fix = fixer.PID(expected_volt, motor_volt) 
                motor_volt = motor_volt_temp
        sleep(0.5)
        self.shoot()
        self.gun_angle = thetaX + fix
        return
    
    def aim_and_fire_target_3(self, target):
        # TODO: (ayala) change this method so it gets global coordinates
        P_ERROR = -75
        I_ERROR = -30
        D_ERROR = 0
        fix = 0
        
        fixer = pid.PID(P_ERROR, I_ERROR, D_ERROR)

        thetaX, expected_volt = fit.bilerp(*target)
        self.rotate(thetaX)

        start_time = time.time()
        # Run the loop for 1 second
        # TODO - change time to global var
        # TODO: get rid of excess delay - in PID end condition and time.sleep. There might be some necessary delay
        motor_volt = self.get_voltage()
        if motor_volt is None:
            motor_volt = 0
        while time.time() - start_time < 1 and np.abs(expected_volt - motor_volt) > PRECISION:
            self.rotate(thetaX + fix)
            motor_volt_temp = self.get_voltage()
            if motor_volt_temp is not None:
                fix = fixer.PID(expected_volt, motor_volt) 
                motor_volt = motor_volt_temp
        sleep(0.5)
        self.shoot()
        self.gun_angle = thetaX + fix
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
