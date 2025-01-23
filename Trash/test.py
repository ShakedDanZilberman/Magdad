from pyfirmata import Arduino, util
from time import sleep

gun_pin = 4
servo_pin = 10
sleep_duration = 0.2
board = Arduino("COM3")
it = util.Iterator(board)
it.start()
servo = board.get_pin(f"d:{servo_pin}:s")
servo.write(20)