from abc import ABC, abstractmethod
import cv2
import numpy as np

from constants import IMG_WIDTH, IMG_HEIGHT

class Handler(ABC):
    """
    Abstract class for image handling.

    A handler is used to store an image and perform operations on it,
    such as displaying it or getting it for the target detection.

    Methods:
        add: Add a frame to the handler. Call this every frame to update the handler.
        get: Get the image from the handler. This returns a black-and-white image, where white pixels represent the suspected targets.
        display: Display the image stored in the handler. This uses cv2.imshow() to display the image with a unique title.
        clear: Clear the image and memory stored in the handler. This is useful for resetting the handler.
    """
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

    def __init__(self, text=None):
        self.img = None
        self.text = text

    def add(self, img):
        std = np.std(img)
        print("RawHandler: add with std:", std)
        self.img = img

    def get(self):
        return self.img

    def display(self, index=0):
        if self.img is None:
            return
        if self.text is not None:
            textcolor = (130, 255, 0)
            cv2.putText(self.img, self.text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1)
        # print(self.img)
        cv2.imshow(RawHandler.TITLE+str(index), self.img)

    def clear(self):
        self.img = None


class ImageParse:
    """
    Class containing utilities for image processing
    
    Methods:
        toGrayscale: Convert an image to grayscale
        differenceImage: Calculate the difference between two images
        blurImage: Blur an image using a Gaussian filter
        aboveThreshold: Apply a threshold to an image, converting it to a binary image
    """
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
            np.ndarray: The thresholded image as a binary image
        """
        return cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)[1]
    
    @staticmethod
    def image_sum(img):
        """Calculate the sum of the pixel values in an image

        Args:
            img (np.ndarray): The image to sum

        Returns:
            int: The sum of the pixel values
        """
        return np.sum(img)  
    
    @staticmethod
    def increase_contrast(img, factor):
        """Increase the contrast of an image

        Args:
            img (np.ndarray): The image to increase the contrast of
            factor (float): The factor to increase the contrast by

        Returns:
            np.ndarray: The image with increased contrast
        """
        return cv2.convertScaleAbs(img, factor, 0)
    
    @staticmethod
    def resize_proportionally(img, factor, timestep: int=0):
        # Get original dimensions
        if img is None:
            return
        (h, w) = img.shape[:2]
        new_width = int(w*factor)
        new_height = int(h * factor)

        # Resize the image
        resized = cv2.resize(img, (new_width, new_height))
        # print(f"Resized image to {new_width}x{new_height} at timestep {timestep}")
        return resized