import cv2
import numpy as np
import constants
from mouseCamera import MouseCameraHandler as mouse_camera

homogrpahy_matrix, status = cv2.findHomography(constants.SRC_ARRAY, constants.DEST_ARRAY, 0, 3)
