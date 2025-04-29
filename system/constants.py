import numpy as np

IMG_WIDTH = 960
IMG_HEIGHT = 540
COM = "COM14"
CAMERA_INDEX = 0
GUN = (0, 0)  # Coordinates of the gun in pixels
# Constants for the homography transformation
H1 = None
H2 = None
H3 = None
FPS = 10

INITIAL_BLURRING_KERNEL = (3, 3)

HIGH_CEP_INDEX = 0.9
LOW_CEP_INDEX = 0.5
# sample rate - in frames, not in seconds
# note - SAMPLE_RATE has to be higher than FRAMES_FOR_INITIALISATION
SAMPLE_RATE = 16
FRAMES_FOR_INITIALISATION = 5
BRIGHTNESS_THRESHOLD = 240

MINIMAL_OBJECT_AREA = 30
MIN_DISTANCE = 0.0

# the following lists are for the homography, specifically used in the calibration in Beit Tzarfat

LENGTH_OF_TABLE = 152
DEST_POINTS = [[0.0, 0.0], [136.0, 0.0], [46.5, 30.0], [105.5, 30.0]]
DEST_ARRAY = np.array(DEST_POINTS, dtype=np.float32)
SRC_POINTS = [[16.0, 92.0], [621.0, 19.0], [134.0, 87.0], [517.0, 45.0]]
SRC_ARRAY = np.array(SRC_POINTS, dtype=np.float32)


# constants for the gun
SLEEP_DURATION = 0.2  # seconds
VOLTAGE_MOTOR_PIN = 4  # pin for the voltage sensor
GUN_PIN = 4  # pin for the gun
SERVO_PIN = 9  # pin for the servo motor
# second calibration - this one worked!
DEST_POINTS_2 = [[0.0, 0.0], [29.0, 0.0], [83.0, 30.0], [136.0, 0.0], [152.0, 0.0], [49.0, 31.0], [104.0, 31.0], [122.0, 48.0], [77.0, 57.0]]
DEST_ARRAY_2 = np.array(DEST_POINTS_2, dtype=np.float32)
SRC_POINTS_2 = [[1.0, 179.0], [163.0, 195.0], [525.0, 231.0], [842.0, 255.0], [935.0, 266.0], [233.0, 246.0], [708.0, 289.0], [936.0, 346.0], [476.0, 341.0]]
SRC_ARRAY_2 = np.array(SRC_POINTS_2, dtype=np.float32)

homography_matrix =  np.array([[-1.91990252e-01, -1.34208894e+00, 2.37769069e+02],
 [ 1.66091588e-01, -1.83397967e+00,  3.20230998e+02],
 [ 1.67233697e-03, -1.70663427e-02,  1.00000000e+00]], dtype=np.float64)

