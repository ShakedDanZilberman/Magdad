import math

from pyfirmata import Arduino, util
from time import sleep
A0 = 0
B0 = 0
C0 = 0
D0 = 90

A1 = 0
B1 = 0
C1 = 0
D1 = 90

# Set up the Arduino board (replace 'COM8' with your Arduino's COM port)
board = Arduino('COM8')  # Adjust the COM port based on your system

# Define the pin for the servo (usually PWM pins)
servoV_pin = 4
servoH_pin = 3# Servo control pin (could be any PWM pin)

# Attach the servo to the board
servoV = board.get_pin(f'd:{servoV_pin}:s')  # 's' means it's a servo
servoH = board.get_pin(f'd:{servoH_pin}:s')
# Start an iterator thread to read analog inputs
it = util.Iterator(board)
it.start()

# Main loop to control the servo
def angle_calc(coordinates):
    X = coordinates[0]
    Y = coordinates[1]
    angleX = D0 + C0*X +B0*X**2 + A0*X**3
    angleY = D1 + C1*X +B1*Y**2 + A1*Y**3
    return angleX, angleY


try:
    while True:
        angleX, angleY = angle_calc([90,90])
        servoH.write(angleX)
        sleep(0.1)
        servoV.write(angleY)
        sleep(0.1)


except KeyboardInterrupt:
    print("Exiting program.")

finally:
    # Clean up and close the board connection
    board.exit()
