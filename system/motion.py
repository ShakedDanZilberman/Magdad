from image_processing import Handler, ImageParse
import cv2


class DifferenceHandler(Handler):
    """
    DifferenceHandler class to calculate the difference between the current frame and the previous frame.
    This is useful for detecting movement.
    IT IS NOT CURRENTLY USED IN THE SYSTEM FOR OBJECT DETECTION.

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
        """Adds the image to the handler. Blurs the difference image to reduce noise."""
        if self.prev is not None:
            self.diff = ImageParse.differenceImage(self.prev, img)
            self.diff = ImageParse.blurImage(self.diff, 20)
        self.prev = img
        return self.diff

    def display(self):
        if self.diff is None:
            return
        cv2.imshow(self.TITLE, self.diff)

    def clear(self):
        raise NotImplementedError("DifferenceHandler does not support clear()")

    def get(self):
        return self.diff

