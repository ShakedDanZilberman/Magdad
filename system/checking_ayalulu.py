import pyfirmata
import time

# Connect to Arduino
port = 'com6'  # Adjust this to your Arduino's port
board = pyfirmata.Arduino(port)

# Start iterator to receive input data
it = pyfirmata.util.Iterator(board)
it.start()

# Define analog pin A0 as an input
analog_pin = board.get_pin('a:0:i')

# Read the input and print the value
try:
    while True:
        analog_value = analog_pin.read()
        if analog_value is not None:
            if voltage <= 0:
                distance = 0
            #Convert the voltage to distance (cm)
            distance = 0.6301 * pow(voltage, -1.17)
            print(f"Analog value: {analog_value:.2f}")
        time.sleep(0.1)
except KeyboardInterrupt:
    # Clean up
    board.exit()
