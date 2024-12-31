import cv2
from image_processing import Handler, ImageParse
import numpy as np

FRAMES_FOR_INITIALISATION = 30
BRIGHTNESS_THRESHOLD = 230

class ChangesHandler(Handler):
    """
    NewPixelsHandler class to accumulate the first N frames and calculate the average.
    It then calculates the difference between the current frame and the average of the first N frames.

    add(img) - adds the image to the handler; if the image is too bright, it is skipped.
                It saves the first N images, and ignores the rest (until clear() is called).
    clear() - clears the images and the average
    get() - returns the average
    display() - displays the difference between the current frame and the average
    """

    first_images = []

    def __init__(self):
        self.images = [None] * FRAMES_FOR_INITIALISATION
        self.index = 0
        self.loading_img = None
        self.title = f"Average of First {FRAMES_FOR_INITIALISATION} Frames"
        self.avg = None

    def add(self, img):
        """Adds the image to the handler."""
        self.img = img

        # Skip if index is -1
        if self.index == -1:
            return

        # Skip images that are too bright
        if np.mean(img) > BRIGHTNESS_THRESHOLD:
            return

        # Add image to the list
        if img is not None:
            self.images[self.index] = img
            self.index += 1

        # Reset index if it exceeds N
        if self.index == FRAMES_FOR_INITIALISATION:
            self.index = -1
            self.get(img)

    def get(self, img):
        """Calculates the average of the first N frames and returns the difference between the current frame and the average of the first N frames"""
        # Return the average if it has already been calculated
        if hasattr(self, "avg") and self.avg is not None:
            diff = ImageParse.differenceImage(img, self.avg)
            diff = ImageParse.blurImage(diff, 20)
            diff = ImageParse.aboveThreshold(diff, 50)
            return diff

        if len(self.images) < FRAMES_FOR_INITIALISATION:
            return None

        # Calculate the average
        self.avg = np.zeros_like(self.images[0])
        # Average only the non-None images
        for i in range(FRAMES_FOR_INITIALISATION):
            # Add the image to the average with a weight of 1/N_effective
            if self.images[i] is not None:
                self.avg = cv2.addWeighted(
                    self.avg, 1, self.images[i], 1 / FRAMES_FOR_INITIALISATION, 0
                )

        return self.avg

    def clear(self):
        """Clears the images and the average, and resets the index"""
        self.images = [None] * FRAMES_FOR_INITIALISATION
        self.avg = None
        self.index = 0

    def isReady(self):
        """Returns True if the average of the first N frames has been accumulated and calculated"""
        return self.index == -1

    def display(self):
        """Displays the difference between the current frame and the average of the first N frames"""
        TITLE = "Changes from Original"
        if not self.isReady():
            LOADING_IMAGE = np.ones(self.img.shape, np.uint8) * 128
            cv2.imshow(TITLE, LOADING_IMAGE)
        else:
            cv2.imshow(TITLE, self.get(self.img))
