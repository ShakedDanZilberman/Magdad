from abc import ABC, abstractmethod
import cv2
import numpy as np

from constants import IMG_WIDTH, IMG_HEIGHT

class Handler(ABC):
    @abstractmethod
    def add(self, img):
        """
        Add an image to the handler.

        Args:
            img (np.ndarray): The image to add to the handler.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get(self):
        """
        Get the image from the handler.

        Returns:
            np.ndarray: The image from the handler.
        """
        pass

    @abstractmethod
    def display(self):
        """
        Display the image stored in the handler.
        Uses cv2.imshow() to display the image.

        Returns:
            None
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear the image and memory stored in the handler.
        """
        pass


class RawHandler(Handler):
    """
    RawHandler class to present the raw image from the camera.
    It does not manipulate the image in any way.
    It is a basic handler for the raw image.

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

    def display(self):
        if self.img is None:
            return
        cv2.imshow(RawHandler.TITLE, self.img)

    def clear(self):
        self.img = None


class ImageParse:
    """Class containing utilities for image processing"""
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
        """Blur an image using a Gaussian filter.

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
    
    @staticmethod
    def image_sum(img):
        return np.sum(img)  
