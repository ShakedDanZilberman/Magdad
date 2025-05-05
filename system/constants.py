import numpy as np

IMG_WIDTH = 960
IMG_HEIGHT = 540
COM = "COM7"
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
MIN_DISTANCE = 3.0


# constants for the gun
SLEEP_DURATION = 0.2  # seconds
VOLTAGE_MOTOR_PIN = 4  # pin for the voltage sensor
GUN_PIN = 4  # pin for the gun
SERVO_PIN = 9  # pin for the servo motor



# constatns for the camera homography
LENGTH_OF_TABLE = 152
# first camera, on blue stand
CAMERA_INDEX_0 = 1
CAMERA_LOCATION_0 = (0, 0)  # Coordinates of the camera in real world
DEST_POINTS_0 = [[0.0, 0.0], [29.0, 0.0], [83.0, 0.0], [136.0, 0.0], [152.0, 0.0], [49.0, 31.0], [104.0, 31.0], [30.0, 48.0], [77.0, 57.0]]
DEST_ARRAY_0 = np.array(DEST_POINTS_0, dtype=np.float32)
SRC_POINTS_0 = [[27.0, 255.0], [294.0, 222.0], [639.0, 188.0], [832.0, 169.0], [879.0, 169.0], [573.0, 269.0], [873.0, 226.0], [480.0, 375.0], [937.0, 326.0]]
SRC_ARRAY_0 = np.array(SRC_POINTS_0, dtype=np.float32)

homography_matrix = np.array([[ 7.71621869e-01,  1.21327031e-01, -4.78733384e+01],
 [ 3.20962251e-01,  3.34565669e+00, -8.39524063e+02],
 [-2.58823543e-03,  3.32245637e-02,  1.00000000e+00]], dtype=np.float64)


# 2 is the closest to the window
CAMERA_INDEX_1 = 2
CAMERA_LOCATION_1 = (0, 0)  # Coordinates of the camera in real world
DEST_POINTS_1 = [[0.0, 0.0], [29.0, 0.0], [83.0, 0.0], [136.0, 0.0], [152.0, 0.0], [49.0, 31.0], [104.0, 31.0], [122.0, 48.0], [77.0, 57.0]]
DEST_ARRAY_1 = np.array(DEST_POINTS_1, dtype=np.float32)
SRC_POINTS_1 = [[1.0, 179.0], [163.0, 195.0], [525.0, 231.0], [842.0, 255.0], [935.0, 266.0], [233.0, 246.0], [708.0, 289.0], [936.0, 346.0], [476.0, 341.0]]
SRC_ARRAY_1 = np.array(SRC_POINTS_1, dtype=np.float32)

homography_matrix_1 =  np.array([[-1.91990252e-01, -1.34208894e+00, 2.37769069e+02],
 [ 1.66091588e-01, -1.83397967e+00,  3.20230998e+02],
 [ 1.67233697e-03, -1.70663427e-02,  1.00000000e+00]], dtype=np.float64)




homography_matrices = [homography_matrix, homography_matrix_1]

