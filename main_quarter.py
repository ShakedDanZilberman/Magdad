# Get all connected devices

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pyfirmata import Arduino, util
import threading

CAMERA_INDEX = 1
MAX_CAMERAS = 10
timestep = 0
centers = [[90, 90]]
# Parameters to play with in calibration
INITIAL_BLURRING_KERNEL = (3, 3)
EDGE_DETECTION_MINTHRESH = 100
EDGE_DETECTION_MAXTHRESH = 130

HIGH_CEP_INDEX = 0.5
LOW_CEP_INDEX = 0.2
DILATION_ITERATIONS = 2
EROSION_ITERATIONS = 5

DILATION_KERNEL = (5, 5)
OPENING_KERNEL = (9, 9)
CLOSING_KERNEL = (5, 5)
EROSION_KERNEL = (3, 3)

CONTOUR_EXTRACTION_M0DE = cv2.RETR_EXTERNAL
CONTOUR_EXTRACTION_METHOD = cv2.CHAIN_APPROX_SIMPLE
CONTOUR_THICKNESS = cv2.FILLED
CONTOUR_HEATMAP_BLUR_KERNEL = (15, 15)
CONTOUR_HEATMAP_STDEV = 15

IMG_WIDTH = 240
IMG_HEIGHT = 320


class Handler(ABC):
    @abstractmethod
    def add(self, img):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def clear(self):
        pass


class ContoursHandler(Handler):
    def __init__(self):
        self.static = None

    def optimize_edges(self, edges):
        # TODO: ADD DOCSTRING
        dilation_kernel = np.ones(DILATION_KERNEL, np.uint8)
        opening_kernel = np.ones(OPENING_KERNEL, np.uint8)
        closing_kernel = np.ones(CLOSING_KERNEL, np.uint8)
        erosion_kernel = np.ones(EROSION_KERNEL, np.uint8)
        dilated_edges = cv2.dilate(
            edges, dilation_kernel, iterations=DILATION_ITERATIONS
        )
        opening = cv2.morphologyEx(dilated_edges, cv2.MORPH_OPEN, opening_kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)
        return cv2.erode(closing, erosion_kernel, iterations=EROSION_ITERATIONS)

    def add(self, img):
        gray = ImageParse.toGrayscale(img)
        height, width = img.shape
        black_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # manipluate the image to get the contours
        blurred = cv2.GaussianBlur(gray, INITIAL_BLURRING_KERNEL, 0)
        edges = cv2.Canny(blurred, EDGE_DETECTION_MINTHRESH, EDGE_DETECTION_MAXTHRESH)
        optimized = self.optimize_edges(edges)
        contours, hierarchy = cv2.findContours(
            optimized, CONTOUR_EXTRACTION_M0DE, CONTOUR_EXTRACTION_METHOD
        )
        cv2.drawContours(black_canvas, contours, -1, (255, 255, 255), CONTOUR_THICKNESS)
        binary_heat_map = ImageParse.toGrayscale(black_canvas)
        # heat_map = cv2.GaussianBlur(black_canvas, CONTOUR_HEATMAP_BLUR_KERNEL, CONTOUR_HEATMAP_STDEV)
        self.static = binary_heat_map

    def display(self, img):
        TITLE = "Contours"
        if self.static is None:
            return
        cv2.imshow(TITLE, self.static)

    def clear(self):
        pass

    def get(self):
        return self.static


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

    # def add_static(self, img, number_of_frames_so_far):
    #     if self.accumulator is None:
    #         self.accumulator = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    #     self.accumulator = cv2.addWeighted(self.accumulator, 1-1/number_of_frames_so_far, img, 1/number_of_frames_so_far, 0)

    def clear(self, img):
        self.accumulator = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        self.window = []

    def get(self):
        return self.accumulator


class RawHandler(Handler):
    """
    RawHandler class to present the raw image from the camera.
    It does not manipulate the image in any way.

    add(img) - adds the image to the handler
    clear() - clears the image
    get() - returns the image
    display() - displays the image
    """

    TITLE = "Camera Connection"

    def __init__(self):
        self.img = None

    def add(self, img):
        self.img = img

    def get(self):
        return self.img

    def display(self, img):
        if self.img is None:
            return
        cv2.imshow(RawHandler.TITLE, img)

    def clear(self):
        self.img = None


class NewPixelsHandler(Handler):
    """
    NewPixelsHandler class to accumulate the first N frames and calculate the average.
    It then calculates the difference between the current frame and the average of the first N frames.

    add(img) - adds the image to the handler; if the image is too bright, it is skipped.
                It saves the first N images, and ignores the rest (until clear() is called).
    clear() - clears the images and the average
    get() - returns the average
    display() - displays the difference between the current frame and the average
    """

    N = 30
    first_N_images = []

    def __init__(self):
        self.images = [None] * NewPixelsHandler.N
        self.index = 0
        self.BRIGHTNESS_THRESHOLD = 230
        self.loading_img = None
        self.title = f"Average of First {NewPixelsHandler.N} Frames"
        self.avg = None

    def add(self, img):
        # Skip if index is -1
        if self.index == -1:
            self.img = img
            return

        # Skip images that are too bright
        if np.mean(img) > self.BRIGHTNESS_THRESHOLD:
            return

        # Add image to the list
        self.images[self.index] = img
        self.index += 1 

        # Reset index if it exceeds N
        if self.index == NewPixelsHandler.N:
            self.index = -1
            self.get(img)
            # cv2.imshow(self.title, self.avg)

    def get(self, img):
        # Return the average if it has already been calculated
        if hasattr(self, "avg") and self.avg is not None:
            diff = ImageParse.differenceImage(img, self.avg)
            diff = ImageParse.blurImage(diff, 20)
            diff = ImageParse.aboveThreshold(diff, 50)
            return diff

        if len(self.images) < NewPixelsHandler.N:
            return None

        # Calculate the average
        self.avg = np.zeros_like(self.images[0])
        # Average only the non-None images
        N_effective = NewPixelsHandler.N - len([i for i in self.images if i is None])
        for i in range(NewPixelsHandler.N):
            # Add the image to the average with a weight of 1/N_effective
            if self.images[i] is not None:
                self.avg = cv2.addWeighted(
                    self.avg, 1, self.images[i], 1 / N_effective, 0
                )

        return self.avg

    def clear(self):
        # Clear the images and the average, and reset the index
        self.shape = self.images[0].shape
        self.images = [None] * NewPixelsHandler.N
        self.avg = None
        self.index = 0

        self.loading_img = np.ones(self.shape, np.uint8) * 128

        # Show the loading image
        # cv2.imshow(self.title, self.loading_img)

    def isReady(self):
        return self.index == -1

    def display(self, img):
        TITLE = "Difference from Original"
        # TITLE2 = 'Cumulative Difference'
        LOADING_IMAGE = np.ones(img.shape, np.uint8) * 128
        if not self.isReady():
            cv2.imshow(TITLE, LOADING_IMAGE)
        else:
            

            # self.cumulative = cv2.addWeighted(self.cumulative, 0.9, diff, 0.1, 0)

            # convert image to RBG
            # diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            # find objects in the image
            # diff = ImageParse.find_objects(diff)  # method redundant
            cv2.imshow(TITLE, self.get(img))
            # cv2.imshow(TITLE2, self.cumulative)


class DifferenceHandler(Handler):
    """
    DifferenceHandler class to calculate the difference between the current frame and the previous frame.
    This is useful for detecting movement.

    add(img) - adds the image to the handler; if the previous image is None, it is skipped.
    clear() - clears the previous image and the difference
    get() - returns the difference
    display() - displays the difference
    """

    TITLE = "Difference"

    def __init__(self):
        self.prev = None
        self.diff = None

    def add(self, img):
        if self.prev is not None:
            self.diff = ImageParse.differenceImage(self.prev, img)
            self.diff = ImageParse.blurImage(self.diff, 20)
        self.prev = img
        return self.diff

    def display(self, img):
        if self.diff is None:
            return
        cv2.imshow(self.TITLE, self.diff)

    def clear(self):
        raise NotImplementedError("DifferenceHandler does not support clear()")

    def get(self):
        return self.diff


class Rendering:
    @staticmethod
    def showMultipleFrames(imgs, titles=None, title=None):
        """
        Display multiple images in a grid using matplotlib.

        Args:
            imgs (list[np.ndarray]): List of images to display
            titles (_type_, optional): _description_. Defaults to None.
            title (_type_, optional): _description_. Defaults to None.
        """
        N = len(imgs)
        # find the optimal p,q such that p*q >= N and p-q is minimized
        p = int(np.ceil(np.sqrt(N)))
        q = int(np.ceil(N / p))
        # Create a grid of p x q subplots
        fig, axs = plt.subplots(p, q, figsize=(10, 7))
        # Remove axis from all subplots
        for ax in axs.flat:
            ax.axis("off")
        # Show the images in the subplots
        for i in range(N):
            ax = axs[i // q, i % q]
            if imgs[i] is not None:
                ax.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            # Set the title of the subplot
            if titles is not None:
                ax.set_title(titles[i])
        # Set the title of the entire figure
        if title is not None:
            plt.suptitle(title)
        plt.show()


class ImageParse:
    @staticmethod
    def toGrayscale(img):
        """Convert an image to grayscale

        Args:
            img (np.ndarray): The image to convert

        Returns:
            np.ndarray: The grayscale image
        """
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return img

    @staticmethod
    def differenceImage(img1, img2):
        """Calculate the difference between two images

        Args:
            img1 (np.ndarray): The first image
            img2 (np.ndarray): The second image

        Returns:
            np.ndarray: The difference between the two images
        """
        # Calculate difference
        if (isinstance(img1, np.ndarray) and img1.size > 1) and (
            isinstance(img2, np.ndarray) and img2.size > 1
        ):
            diff = cv2.absdiff(img1, img2)
            # Apply threshold
            _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            return diff
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

    @staticmethod
    def blurImage(img, factor=5):
        """Blur an image using a Gaussian filter

        Args:
            img (np.ndarray): The image to blur
            factor (int, optional): The size of the filter. Defaults to 5.

        Returns:
            np.ndarray: The blurred image
        """
        if factor % 2 == 0:
            factor += 1
        return cv2.GaussianBlur(img, (factor, factor), 0)

    @staticmethod
    def aboveThreshold(img, threshold):
        """Apply a threshold to an image

        Args:
            img (np.ndarray): The image to threshold
            threshold (int): The threshold value

        Returns:
            np.ndarray: The thresholded image
        """
        return cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)[1]

    # @staticmethod
    # def find_objects(img):
    # # TODO: this is a horrible way to find objects, please improve
    # # This function is supposed to find objects in the image
    # # The image is a black and white image with white blobs on a black background
    # params = cv2.SimpleBlobDetector_Params()
    # params.maxThreshold = 255
    # params.minThreshold = 10
    # # Filter by color (white blobs)
    # params.filterByColor = True
    # params.blobColor = 255

    # # Filter by area
    # params.filterByArea = True
    # params.minArea = 40  # Minimum area of blob
    # params.maxArea = 10000  # Maximum area of blob
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(img)

    # # Get the brightness of each blob
    # brightnesses = ImageParse.evaluateBrightness(img, keypoints)

    # img = ImageParse.drawKeypointsOnImage(img, keypoints, brightnesses)

    # return img

    @staticmethod
    def evaluateBrightness(img, keypoints):
        """Evaluate the brightness of each keypoint in an image

        Args:
            img (np.ndarray): The image to evaluate
            keypoints (list): List of keypoints

        Returns:
            dict: A dictionary containing the brightness of each keypoint
        """
        radii = {keypoint: int(keypoint.size / 2) for keypoint in keypoints}
        bounding_boxes = {
            keypoint: (
                int(keypoint.pt[0] - keypoint.size / 2),
                int(keypoint.pt[1] - keypoint.size / 2),
                radius,
                radius,
            )
            for keypoint, radius in radii.items()
        }
        for keypoint, (x, y, w, h) in bounding_boxes.items():
            bounding_boxes[keypoint] = (
                max(0, x),
                max(0, y),
                min(img.shape[1], w),
                min(img.shape[0], h),
            )
        brightnesses = {
            keypoint: np.mean(img[y : y + h, x : x + w])
            for keypoint, (x, y, w, h) in bounding_boxes.items()
        }
        return brightnesses

    @staticmethod
    def drawKeypointsOnImage(img, keypoints, brightnesses, cross_size=2):
        # Convert the image to color
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        centerpoints = {
            keypoint: (int(keypoint.pt[0]), int(keypoint.pt[1]))
            for keypoint in keypoints
        }
        radii = {keypoint: int(keypoint.size / 2) for keypoint in keypoints}
        # draw the keypoints
        for keypoint in keypoints:
            # Draw a circle around the blob
            x, y = centerpoints[keypoint]
            radius = radii[keypoint]
            brightness = brightnesses[keypoint]
            # color is a lerp between green and blue, where blue is for low brightness and green is for high brightness
            color = (0, brightness, int(255 - brightness))
            cv2.circle(img, (x, y), radius, color, 2)
            cv2.circle(img, (x, y), radius, (0, int(brightness), 0), -1)
            # Draw a cross on the blob
            cv2.line(img, (x - cross_size, y), (x + cross_size, y), color, 1)
            cv2.line(img, (x, y - cross_size), (x, y + cross_size), color, 1)

        return img

    @staticmethod
    def generate_targets(heat_map: cv2.typing.MatLike):
        """Generate targets from a heatmap.

        Args:
            heat_map (cv2.typing.MatLike): The heatmap to generate targets from

        Returns:
            Tuple: A tuple containing the targets for CEP_HIGH and CEP_LOW
        """
        high_intensity = int(HIGH_CEP_INDEX * 255)
        low_intensity = int(LOW_CEP_INDEX * 255)
        _, reduction_high = cv2.threshold(
            heat_map, high_intensity - 1, high_intensity, cv2.THRESH_BINARY
        )
        _, reduction_low = cv2.threshold(
            heat_map, low_intensity - 1, low_intensity, cv2.THRESH_BINARY
        )
        reduction_high = cv2.GaussianBlur(reduction_high, INITIAL_BLURRING_KERNEL, 0)
        reduction_low = cv2.GaussianBlur(reduction_low, INITIAL_BLURRING_KERNEL, 0)
        CEP_HIGH = cv2.Canny(reduction_high, 100, 150)
        CEP_LOW = cv2.Canny(reduction_low, 127, 128)
        contours_high, _ = cv2.findContours(
            CEP_HIGH, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        contours_low, _ = cv2.findContours(
            CEP_LOW, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        high_targets = []
        low_targets = []
        contour_centers = []
        for contour in contours_high:
            # add accurate CEP to list
            (x, y), radius = cv2.minEnclosingCircle(contour)
            new_circle = (x, y), radius
            high_targets.append(new_circle)
            # add contour center to list
            M = cv2.moments(contour)
            if not M["m00"] == 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                contour_centers.append((int(cx), int(cy)))
            else:
                contour_centers.append((int(x), int(y)))
        for contour in contours_low:
            # add inaccurate CEP to list
            (x, y), radius = cv2.minEnclosingCircle(contour)
            new_circle = (x, y), radius
            low_targets.append(new_circle)
        return high_targets, low_targets, contour_centers


class CameraIO:
    def detectCameras():
        """Detect all connected cameras and display their images in a grid using matplotlib

        If the camera at CAMERA_INDEX is connected, the function will print a message and return.
        """
        # first try to connect to CAMERA_INDEX
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if cam.isOpened():
            print(f"Camera @ index {CAMERA_INDEX} is connected")
            cam.release()
            return
        # Otherwise, try to connect to all cameras
        imgs = [None] * MAX_CAMERAS
        for i in range(MAX_CAMERAS):
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                print(f"Camera @ index {i} is connected")
                # Get a frame from the camera
                ret_val, imgs[i] = cam.read()
                cam.release()
        # Show all images in matplotlib window
        Rendering.showMultipleFrames(
            imgs,
            [f"Camera {i}" for i in range(MAX_CAMERAS)],
            f"Failed to connect to camera #{CAMERA_INDEX}\nAll Available Cameras",
        )
        plt.show()

    def saveImage(img, path):
        """Save an image to a file

        Args:
            img (np.ndarray): The image to save
            path (str): The path to save the image to
        """
        cv2.imwrite(path, img)


class LaserPointer:
    def __init__(self):
        self.point = (0, 0)
        self.board = Arduino("COM6")

        # Define the pins for the servos and the laser
        servoV_pin = 5
        servoH_pin = 3
        laser_pin = 8
        self.board.digital[laser_pin].write(1)
        # Attach the servo to the board
        self.servoV = self.board.get_pin(f"d:{servoV_pin}:s")  # 's' means it's a servo
        self.servoH = self.board.get_pin(f"d:{servoH_pin}:s")

        # Start an iterator thread to read analog inputs
        it = util.Iterator(self.board)
        it.start()
        # Coefficients for angleX polynomial
        self.AX = 113.9773
        self.BX = -0.0588
        self.CX = 0.0001
        self.DX = 0
        self.EX = -0.1736
        self.FX = 0.0001
        self.GX = 0.0000
        self.HX = 0.0001
        self.IX = -0.0000
        self.JX = 0

        # Coefficients for angleY polynomial
        self.AY = 69.3912
        self.BY = -0.1502
        self.CY = 0.0001
        self.DY = 0
        self.EY = 0.0144
        self.FY = 0.0
        self.GY = 0.0000
        self.HY = 0.0000
        self.IY = 0.0000
        self.JY = 0

        self.STARTX = 60
        self.STARTY = 40
        self.deltaX = 30
        self.deltaY = 20
        self.NUMITER = 10

    def calibration():
        mx, my = 0, 0
        def click_event(event, x, y, flags, param):
            global mx, my
            if event == cv2.EVENT_LBUTTONDOWN:
                # print(f"Clicked coordinates: {relative_x}, {relative_y}")
                mx, my = x, y
        def find_red_point(frame):
            """
            Finds the (x, y) coordinates of the single red point in the image.

            Args:
                frame: The input image (BGR format).

            Returns:
                A tuple (x, y) representing the center of the red point if found, or None if no red point is detected.
            """
            # Convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for red in HSV
            lower_red1 = np.array([0, 120, 70])  # Lower range for red
            upper_red1 = np.array([10, 255, 255])  # Upper range for red
            lower_red2 = np.array([170, 120, 70])  # Lower range for red (wrapping around 180 degrees)
            upper_red2 = np.array([180, 255, 255])  # Upper range for red

            # Create masks for red
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)  # Combine masks

            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return (0, 0)  # No red point found

            # Assume the largest contour is the red point (adjust as needed)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the center of the red point
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return (0, 0)  # Avoid division by zero

            cX = int(M["m10"] / M["m00"])  # x-coordinate
            cY = int(M["m01"] / M["m00"])  # y-coordinate

            return cX, cY
        
    def angle_calc(self, coordinates):
        X = coordinates[0]  # rx
        Y = coordinates[1]  # ry
        # Calculate angleX using the full polynomial expression
        angleX = (
            self.AX
            + self.BX * Y
            + self.CX * Y**2
            + self.DX * Y**3
            + self.EX * X
            + self.FX * X * Y
            + self.GX * X * Y**2
            + self.HX * X**2
            + self.IX * X**2 * Y
            + self.JX * X**3
        )
        # Calculate angleY using the full polynomial expression
        angleY = (
            self.AY
            + self.BY * Y
            + self.CY * Y**2
            + self.DY * Y**3
            + self.EY * X
            + self.FY * X * Y
            + self.GY * X * Y**2
            + self.HY * X**2
            + self.IY * X**2 * Y
            + self.JY * X**3
        )
        return angleX, angleY

    def move(self, point):
        self.point = point
        angleX, angleY = self.angle_calc(point)
        print(angleX, angleY)
        self.servoH.write(angleX)
        self.servoV.write(angleY)


class DecisionMaker:
    @staticmethod
    def avg_heat_maps(changes_map, contours_map):
        """Intersect two heatmaps

        Args:
            heatmap1 (np.ndarray): The first heatmap
            heatmap2 (np.ndarray): The second heatmap

        Returns:
            np.ndarray: The intersection of the two heatmaps
        """
        
        if (isinstance(changes_map, np.ndarray) and changes_map.size > 1) and (
            isinstance(contours_map, np.ndarray) and contours_map.size > 1
        ):
            if np.mean(changes_map) < 20:
                return contours_map
            if np.mean(contours_map) < 20:
                return changes_map
            result = changes_map + contours_map
            result = np.clip(result, 0, 255)
            return result
        if isinstance(changes_map, np.ndarray) and changes_map.size > 1:
            return changes_map
        if isinstance(contours_map, np.ndarray) and contours_map.size > 1:
            return contours_map
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

def show_targets(average):
    average = cv2.cvtColor(average, cv2.COLOR_GRAY2BGR)
    circles_high, circles_low, centers = ImageParse.generate_targets(average)
    for circle in circles_low:
        cv2.circle( 
            average,
            (int(circle[0][0]), int(circle[0][1])),
            int(circle[1]),
            (0, 255, 0),
            1,
        )
    for circle in circles_high:
        cv2.circle(
            average,
            (int(circle[0][0]), int(circle[0][1])),
            int(circle[1]),
            (0, 0, 255),
            1,
        )
    for center in centers:
        cv2.circle(average, center, radius=1, color=(255, 0, 0), thickness=-1)
    cv2.imshow("average", average)
    return circles_high, circles_low, centers


def laser_thread():
    global centers
    laser_pointer = LaserPointer()
    while True:
        for center in centers:
            laser_pointer.move()
            time.sleep(0.3)


def main():
    global CAMERA_INDEX, timestep, centers
    CameraIO.detectCameras()
    cam = cv2.VideoCapture(CAMERA_INDEX)
    rawHandler = RawHandler()
    newPixelsHandler = NewPixelsHandler()
    differenceHandler = DifferenceHandler()
    contoursHandler = ContoursHandler()
    accumulator_contours = Accumulator()
    laser = threading.Thread(target=laser_thread)
    laser.start()

    
    number_of_frames = 0
    while True:
        number_of_frames += 1
        ret_val, img = cam.read()
        img = ImageParse.toGrayscale(img)

        if not ret_val:
            print("Camera @ index 1 not connected")
            CAMERA_INDEX = int(
                input("Enter the index of the camera you want to connect to: ")
            )
            cam = cv2.VideoCapture(CAMERA_INDEX)
            ret_val, img = cam.read()
            if not ret_val:
                print("Failed to connect to camera")
                break

        # Downsize and grayscale the image to better simulate IR camera - Remove this when IR camera is connected
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = ImageParse.toGrayscale(img)

        for handler in [
            rawHandler,
            newPixelsHandler,
            differenceHandler,
            contoursHandler,
        ]:
            handler.add(img)
            handler.display(img)
        
        # accumulator_changes.add(newPixelsHandler.get(), 0.9, 0.1)
        # changes_heat_map = newPixelsHandler.get()
        accumulator_contours.add(contoursHandler.get(), 0.99, 0.01)
        # show accumulated heatmaps
        # cv2.imshow("acc changes", accumulator_changes.get())
        cv2.imshow("acc contours", accumulator_contours.get())

        average = DecisionMaker.avg_heat_maps(newPixelsHandler.get(img), contoursHandler.get())
        cv2.imshow("newPixelsHandler.get()", newPixelsHandler.get(img))
        cv2.imshow("average before", average)
        circles_high, circles_low, centers = show_targets(average=average)
        
        if cv2.waitKey(1) == 32:  # Whitespace
            newPixelsHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
