import cv2
from image_processing import Handler, ImageParse
import numpy as np
from constants import IMG_HEIGHT, IMG_WIDTH, FRAMES_FOR_INITIALISATION
BRIGHTNESS_THRESHOLD = 240

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
        self.frames_remaining_to_initialize = FRAMES_FOR_INITIALISATION

    def add(self, img):
        """Adds the image to the handler."""
        self.img = img

        # Skip if index is -1
        if self.index != -1:
            # Skip images that are too bright
            if img is None:
                return
            if np.mean(img) > BRIGHTNESS_THRESHOLD:
                return

            # Add image to the list
            if img is not None:
                self.images[self.index] = img
                self.index += 1

            # Reset index if it exceeds N
            if self.index < FRAMES_FOR_INITIALISATION:
                return
            
            self.index = -1
            self.avg = np.zeros_like(self.images[0])
            for i in range(FRAMES_FOR_INITIALISATION):
                self.avg = cv2.addWeighted(self.avg, 1, self.images[i], 1 / FRAMES_FOR_INITIALISATION, 0)
        
        self.diff = ImageParse.differenceImage(img, self.avg)
        self.diff = ImageParse.blurImage(self.diff, 20)
        self.diff = ImageParse.aboveThreshold(self.diff, 10)
        # self.diff = ImageParse.increase_contrast(self.diff, 1.5)  # leave this commented out for now, until it's fixed


    def get(self):
        """Calculates the average of the first N frames and returns the difference between the current frame and the average of the first N frames"""
        # Return the average if it has already been calculated
        if hasattr(self, "diff") and self.diff is not None:
            return self.diff

        if len(self.images) < FRAMES_FOR_INITIALISATION:
            return None

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
            LOADING_IMAGE = np.ones((IMG_HEIGHT, IMG_WIDTH), np.uint8) * 128
            cv2.imshow(TITLE, LOADING_IMAGE)
        else:
            cv2.imshow(TITLE, self.diff)


class NewChangesHandler(Handler):
    """
    ChangesHandler class to accumulate the first N valid frames and calculate the average.
    It calculates the difference between the current frame and the average of those frames.
    """

    def __init__(self):
        self.images = []
        self.index = 0
        self.valid_count = 0
        self.avg = None
        self.diff = None
        self.title = f"Average of First {FRAMES_FOR_INITIALISATION} Frames"

    def add(self, img):
        """Adds the image to the handler."""

        if img is None:
            print("Skipped: Image is None")
            return

        mean_brightness = np.mean(img)
        if mean_brightness > BRIGHTNESS_THRESHOLD:
            print(f"Skipped: Bright frame (mean = {mean_brightness:.2f})")
            return

        if self.valid_count < FRAMES_FOR_INITIALISATION:
            self.images.append(img)
            self.valid_count += 1
            print(f"Added frame {self.valid_count}/{FRAMES_FOR_INITIALISATION} (mean brightness = {mean_brightness:.2f})")

            if self.valid_count == FRAMES_FOR_INITIALISATION:
                print("Initialization complete: computing average image...")
                self.avg = np.zeros_like(self.images[0], dtype=np.float32)
                for i in range(FRAMES_FOR_INITIALISATION):
                    self.avg = cv2.addWeighted(self.avg, 1, self.images[i].astype(np.float32), 1 / FRAMES_FOR_INITIALISATION, 0)
                self.avg = self.avg.astype(np.uint8)

            return  # Don't compute diff until initialized

        # Once initialized, calculate diff
        self.diff = ImageParse.differenceImage(img, self.avg)
        self.diff = ImageParse.blurImage(self.diff, 20)
        self.diff = ImageParse.aboveThreshold(self.diff, 10)

    def get(self):
        """Returns the difference image if ready."""
        return self.diff if self.isReady() else None

    def clear(self):
        """Clears the accumulated images and resets the handler."""
        print("changes are being cleared")
        self.images.clear()
        self.valid_count = 0
        self.avg = None
        self.diff = None

    def isReady(self):
        """Checks if the average has been computed."""
        return self.valid_count >= FRAMES_FOR_INITIALISATION and self.avg is not None

    def display(self):
        """Displays the current difference image or loading screen."""
        TITLE = "Changes from Original"
        if not self.isReady():
            loading_img = np.ones((IMG_HEIGHT, IMG_WIDTH), np.uint8) * 128
            cv2.imshow(TITLE, loading_img)
        else:
            cv2.imshow(TITLE, self.diff)

