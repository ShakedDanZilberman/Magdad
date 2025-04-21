import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 480
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


