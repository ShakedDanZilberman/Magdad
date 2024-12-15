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

# PID constants (tune these based on your system)
Kp = 1.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.01  # Derivative gain

# Initialize previous values for PID
prev_errorX = 0
prev_errorY = 0
integralX = 0
integralY = 0
dt = 0.01  # Time step (seconds)

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

def calculate_PID_coefficients(errorX, errorY):
    global prev_errorX, prev_errorY, integralX, integralY

    # Proportional term
    P_X = Kp * errorX
    P_Y = Kp * errorY

    # Integral term
    integralX += errorX * dt
    integralY += errorY * dt
    I_X = Ki * integralX
    I_Y = Ki * integralY

    # Derivative term
    derivativeX = (errorX - prev_errorX) / dt
    derivativeY = (errorY - prev_errorY) / dt
    D_X = Kd * derivativeX
    D_Y = Kd * derivativeY

    # Update previous errors
    prev_errorX = errorX
    prev_errorY = errorY

    # Calculate coefficients
    global A0, B0, C0, D0, A1, B1, C1, D1
    C0 = P_X + I_X + D_X
    B0 = 0  # Modify based on specific requirements
    A0 = 0  # Modify based on specific requirements
    C1 = P_Y + I_Y + D_Y
    B1 = 0  # Modify based on specific requirements
    A1 = 0  # Modify based on specific requirements

    print(A0, B0, C0, D0, A1, B1, C1, D1)



try:
    while True:
        calculate_PID_coefficients(1,1)
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
