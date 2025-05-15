import numpy as np

IMG_WIDTH = 960
IMG_HEIGHT = 540

GUN = (0, 0)  # Coordinates of the gun in pixels

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


# constatns for the camera homography, positions are when looking at the door from inside the room
# left camera
CAMERA_INDEX_0 = 2
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
CAMERA_INDEX_2 = 2
CAMERA_LOCATION_2 = (0, 0)  # Coordinates of the camera in real world
DEST_POINTS_2 = [[175.0, 0.0], [200.0, 0.0], [225.0, 0.0], [250.0, 0.0], [275.0, 0.0],
                 [175.0, 25.0], [200.0, 25.0], [225.0, 25.0], [250.0, 25.0], [275.0, 25.0],
                    [200.0, 50.0], [225.0, 50.0], [250.0, 50.0], [275.0, 50.0],
                    [225.0, 75.0], [250.0, 75.0]]
DEST_ARRAY_2 = np.array(DEST_POINTS_1, dtype=np.float32)
SRC_POINTS_2 = [[108.0, 37.0], [258.0, 31.0], [417.0, 27.0], [592.0, 15.0], [782.0, 3.0],
                 [35.0, 95.0], [211.0, 91.0], [402.0, 85.0], [614.0, 75.0], [841.0, 54.0],
                   [138.0, 183.0], [382.0, 182.0], [661.0, 170.0], [959.0, 151.0],
                     [346.0, 355.0], [758.0, 350.0]]
SRC_ARRAY_2 = np.array(SRC_POINTS_1, dtype=np.float32)

homography_matrix_2 =  np.array([[3.47340665e-01, 9.84985803e-01,  3.59704322e+01],
 [1.55899206e-02,  1.11032786e+00, -1.30235758e+02],
 [2.85124685e-04,  8.30222674e-03,  1.00000000e+00]], dtype=np.float64)

homography_matrices = [homography_matrix, homography_matrix_1, homography_matrix_2]


