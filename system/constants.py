import numpy as np

IMG_WIDTH = 960
IMG_HEIGHT = 540
COM = "COM6"
CAMERA_INDEX = 0
GUN = (0, 0)  # Coordinates of the gun in pixels

# Constants for the homography transformation
H1 = None
H2 = None
H3 = None

# the following lists are for the homography, specifically used in the calibration in Beit Tzarfat

LENGTH_OF_TABLE = 152
DEST_POINTS = [[0.0, 0.0], [136.0, 0.0], [46.5, 30.0], [105.5, 30.0]]
DEST_ARRAY = np.array(DEST_POINTS, dtype=np.float32)
SRC_POINTS = [[16.0, 92.0], [621.0, 19.0], [134.0, 87.0], [517.0, 45.0]]
SRC_ARRAY = np.array(SRC_POINTS, dtype=np.float32)


# second calibration - this one worked!
DEST_POINTS_2 = [[0.0, 0.0], [29.0, 0.0], [83.0, 30.0], [136.0, 0.0], [152.0, 0.0], [49.0, 31.0], [104.0, 31.0], [122.0, 48.0], [77.0, 57.0]]
DEST_ARRAY_2 = np.array(DEST_POINTS_2, dtype=np.float32)
SRC_POINTS_2 = [[7.0, 201.0], [167.0, 183.0], [526.0, 147.0], [853.0, 107.0], [957.0, 99.0], [233.0, 223.0], [714.0, 169.0], [952.0, 186.0], [473.0, 272.0]]
SRC_ARRAY_2 = np.array(SRC_POINTS_2, dtype=np.float32)

homography_matrix = np.array([
    [-3.81204087e-01, -1.04290591e+00, 2.09868308e+02],
    [-1.59146621e-01, -1.44444364e+00, 2.83510262e+02],
    [-1.45803869e-03, -1.32499476e-02, 1.00000000e+00]
], dtype=np.float64)

