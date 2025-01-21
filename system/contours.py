from image_processing import ImageParse, Handler
import numpy as np
import cv2
from constants import IMG_WIDTH, IMG_HEIGHT


# These parameters are used to optimize the edges for contour extraction
# Tune these parameters to get the best results

DILATION_KERNEL = (5, 5)
OPENING_KERNEL = (5, 5)
CLOSING_KERNEL = (3, 3)
EROSION_KERNEL = (3, 3)

DILATION_ITERATIONS = 2
EROSION_ITERATIONS = 2

INITIAL_BLURRING_KERNEL = (3, 3)
EDGE_DETECTION_MINTHRESH = 150
EDGE_DETECTION_MAXTHRESH = 180


CONTOUR_EXTRACTION_M0DE = cv2.RETR_EXTERNAL
CONTOUR_EXTRACTION_METHOD = cv2.CHAIN_APPROX_SIMPLE
CONTOUR_THICKNESS = cv2.FILLED
CONTOUR_HEATMAP_BLUR_KERNEL = (15, 15)
CONTOUR_HEATMAP_STDEV = 15

class Kernels:
    dilation_kernel = np.ones(DILATION_KERNEL, np.uint8)
    opening_kernel = np.ones(OPENING_KERNEL, np.uint8)
    closing_kernel = np.ones(CLOSING_KERNEL, np.uint8)
    erosion_kernel = np.ones(EROSION_KERNEL, np.uint8)

class ContoursHandler(Handler):
    def __init__(self):
        self.static = None

    def optimize_edges(self, edges, show=False):
        """
        Optimize the edges for contour extraction.
        Uses dilation, opening, closing, and erosion to optimize the edges.
        Parameters are defined at the top of the file.

        Args:
            edges (np.ndarray): The edges to optimize.

        Returns:
            np.ndarray: The optimized edges.
        """
        dilated_edges = cv2.dilate(
            edges, Kernels.dilation_kernel, iterations=1
        )
        opening = cv2.morphologyEx(dilated_edges, cv2.MORPH_OPEN, Kernels.opening_kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, Kernels.closing_kernel)
        erode = cv2.erode(closing, Kernels.erosion_kernel, iterations=EROSION_ITERATIONS)

        if show:
            cv2.imshow("original edges", edges)
            cv2.imshow("dilation", dilated_edges)
            cv2.imshow("opening", opening)
            cv2.imshow("closing", closing)
            cv2.imshow("erode", erode)
        return erode

    def add(self, img):
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
        gray = ImageParse.toGrayscale(img)
        height, width = img.shape
        black_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # manipluate the image to get the contours
        blurred = cv2.GaussianBlur(gray, INITIAL_BLURRING_KERNEL, 0)
        edges = cv2.Canny(blurred, EDGE_DETECTION_MINTHRESH, EDGE_DETECTION_MAXTHRESH)
        optimized = self.optimize_edges(edges, True)
        contours, hierarchy = cv2.findContours(
            optimized, CONTOUR_EXTRACTION_M0DE, CONTOUR_EXTRACTION_METHOD
        )
        cv2.drawContours(black_canvas, contours, -1, (255, 255, 255), CONTOUR_THICKNESS)
        black_canvas = cv2.erode(black_canvas, Kernels.erosion_kernel, iterations=EROSION_ITERATIONS)
        black_canvas = cv2.dilate(black_canvas, Kernels.dilation_kernel, iterations=DILATION_ITERATIONS)
        binary_heat_map = ImageParse.toGrayscale(black_canvas)
        
        self.static = binary_heat_map

    def display(self):
        TITLE = "Contours"
        if self.static is None:
            return
        cv2.imshow(TITLE, self.static)

    def clear(self):
        pass

    def get(self):
        return self.static



### REDUNDANT CODE ###
class Accumulator:
    """
    Accumulator class to accumulate B&W heatmaps
    It adds images with a weight of 0.1 to the accumulator

    add(img) - adds the image to the accumulator
    clear() - clears the accumulator
    get() - returns the accumulator image
    """
    LOOKBACK = 10

    def __init__(self):
        raise NotImplementedError("This class is redundant and should not be used.")
        self.accumulator = None
        self.window = []

    def add(self, img, prev_weight, new_weight):
        if not isinstance(img, np.ndarray):
            return
        if len(img.shape) < 2:
            return
        if self.accumulator is None:
            self.accumulator = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        self.window.append(img)
        if len(self.window) > Accumulator.LOOKBACK:
            self.window.pop(0)
        moving_average = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for frame in self.window:
            moving_average += (frame * 0.3).astype(np.uint8)
        moving_average = np.clip(moving_average, 0, 255)
        moving_average = moving_average.astype(np.uint8)
        self.accumulator = moving_average

    def clear(self, img):
        self.accumulator = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        self.window = []

    def get(self):
        return self.accumulator