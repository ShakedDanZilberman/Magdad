import numpy as np

IMG_WIDTH = 960
IMG_HEIGHT = 540
COM = "COM18"  # COM port for the Arduino
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
MIN_DISTANCE = 0.5
HISTORY_DELAY = 10

# constants for the gun
SLEEP_DURATION = 0.2  # seconds
VOLTAGE_MOTOR_PIN = 4  # pin for the voltage sensor
GUN_PIN = 4  # pin for the gun
SERVO_PIN = 9  # pin for the servo motor



# # constatns for the camera homography
# LENGTH_OF_TABLE = 152
# # first camera, on blue stand
# CAMERA_INDEX_0 = 1
# CAMERA_LOCATION_0 = (0, 0)  # Coordinates of the camera in real world
# DEST_POINTS_0 = [[0.0, 0.0], [29.0, 0.0], [83.0, 0.0], [136.0, 0.0], [152.0, 0.0], [49.0, 31.0], [104.0, 31.0], [30.0, 48.0], [77.0, 57.0]]
# DEST_ARRAY_0 = np.array(DEST_POINTS_0, dtype=np.float32)
# SRC_POINTS_0 = [[27.0, 255.0], [294.0, 222.0], [639.0, 188.0], [832.0, 169.0], [879.0, 169.0], [573.0, 269.0], [873.0, 226.0], [480.0, 375.0], [937.0, 326.0]]
# SRC_ARRAY_0 = np.array(SRC_POINTS_0, dtype=np.float32)

# homography_matrix = np.array([[ 7.71621869e-01,  1.21327031e-01, -4.78733384e+01],
#  [ 3.20962251e-01,  3.34565669e+00, -8.39524063e+02],
#  [-2.58823543e-03,  3.32245637e-02,  1.00000000e+00]], dtype=np.float64)


# # 2 is the closest to the window
# CAMERA_INDEX_1 = 2
# CAMERA_LOCATION_1 = (0, 0)  # Coordinates of the camera in real world
# DEST_POINTS_1 = [[0.0, 0.0], [29.0, 0.0], [83.0, 0.0], [136.0, 0.0], [152.0, 0.0], [49.0, 31.0], [104.0, 31.0], [122.0, 48.0], [77.0, 57.0]]
# DEST_ARRAY_1 = np.array(DEST_POINTS_1, dtype=np.float32)
# SRC_POINTS_1 = [[1.0, 179.0], [163.0, 195.0], [525.0, 231.0], [842.0, 255.0], [935.0, 266.0], [233.0, 246.0], [708.0, 289.0], [936.0, 346.0], [476.0, 341.0]]
# SRC_ARRAY_1 = np.array(SRC_POINTS_1, dtype=np.float32)

# homography_matrix_1 =  np.array([[-1.91990252e-01, -1.34208894e+00, 2.37769069e+02],
#  [ 1.66091588e-01, -1.83397967e+00,  3.20230998e+02],
#  [ 1.67233697e-03, -1.70663427e-02,  1.00000000e+00]], dtype=np.float64)
# homography_matrices = [homography_matrix, homography_matrix_1]



# constatns for the camera homography, positions are when looking at the door from inside the room
# left camera
CAMERA_INDEX_0 = 1
CAMERA_LOCATION_0 = (0, 0)  # Coordinates of the camera in real world
DEST_POINTS_0 = [[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [100.0, 0.0], 
                 [25.0, 50.0], [50.0, 50.0], [75.0, 50.0]]

DEST_ARRAY_0 = np.array(DEST_POINTS_0, dtype=np.float32)
SRC_POINTS_0 = [[76.0, 68.0], [232.0, 68.0], [397.0, 66.0], [565.0, 62.0], [727.0, 61.0],
                  [125.0, 212.0], [378.0, 212.0], [656.0, 207.0]]
SRC_ARRAY_0 = np.array(SRC_POINTS_0, dtype=np.float32)

homography_matrix = np.array([[0.22884, 0.33274, -39.792],
 [0.010142, 0.8147, -56.995],
 [0.0001245, 0.006266, 1]], dtype=np.float64)


# center camera
CAMERA_INDEX_1 = 0
CAMERA_LOCATION_1 = (0, 0)  # Coordinates of the camera in real world
DEST_POINTS_1 = [[100.0, 0.0], [125.0, 0.0], [150.0, 0.0], [175.0, 0.0], [200.0, 0.0], [125.0, 50.0], [150.0, 50.0], [175.0, 50.0]]

DEST_ARRAY_1 = np.array(DEST_POINTS_1, dtype=np.float32)
SRC_POINTS_1 = [[184.0, 118.0], [332.0, 116.0], [488.0, 108.0], [662.0, 110.0], 
                [844.0, 109.0], [239.0, 256.0], [485.0, 249.0], [774.0, 262.0]]
SRC_ARRAY_1 = np.array(SRC_POINTS_1, dtype=np.float32)

homography_matrix_1 =  np.array([
    [0.50381, 1.655, -44.38],
    [0.022274, 1.453, -174.04],
    [0.00071776, 0.011025, 1]
], dtype=np.float64)


# right camera
CAMERA_INDEX_2 = 3
CAMERA_LOCATION_2 = (0, 0)  # Coordinates of the camera in real world
DEST_POINTS_2 = [[175.0, 0.0], [200.0, 0.0], [225.0, 0.0], [250.0, 0.0], [275.0, 0.0],
                 [175.0, 25.0], [200.0, 25.0], [225.0, 25.0], [250.0, 25.0], [275.0, 25.0],
                    [200.0, 50.0], [225.0, 50.0], [250.0, 50.0], [275.0, 50.0]]
DEST_ARRAY_2 = np.array(DEST_POINTS_2, dtype=np.float32)
SRC_POINTS_2 = [[99.0, 43.0], [251.0, 37.0], [411.0, 29.0], [583.0, 19.0], [769.0, 3.0],
                 [27.0, 99.0], [205.0, 94.0], [397.0, 87.0], [608.0, 75.0], [832.0, 59.0],
                   [133.0, 188.0], [375.0, 185.0], [653.0, 173.0], [959.0, 155.0]]
SRC_ARRAY_2 = np.array(SRC_POINTS_2, dtype=np.float32)

homography_matrix_2 =  np.array([[0.34001,1.2854, 137.03],
                                  [0.04361, 0.73084,-37.098],
                                    [0.00057558,0.0055111, 1]], dtype=np.float64)

homography_matrices = [homography_matrix, homography_matrix_1, homography_matrix_2]


