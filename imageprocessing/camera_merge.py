# Get all connected devices

import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

CAMERA_INDEX = 1
MAX_CAMERAS = 10
image_index = 0


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

    def add(self, img):
        gray = ImageParse.toGrayscale(img)
        # manipluate the image to get the contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 150, 200)
        kernel_small = np.ones((3, 3), np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        kernel_big = np.ones((7, 7), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        opening = cv2.morphologyEx(dilated_edges, cv2.MORPH_OPEN, kernel_big)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        erode = cv2.erode(closing, kernel_small, iterations=4)
        self.static = erode

    def display(self, img):
        TITLE = 'Contours'
        if self.static is None:
            return
        cv2.imshow(TITLE, self.static)

    def clear(self):
        pass

    def get(self):
        return self.static


class Accumulator:
    """
    img is the curent frame in grayscale
    accumulator is the accumulated grayscale img
        """
    def __init__(self):
        self.img = None
        self.accumulator = None

    def add(self, img):
        if self.accumulator is None:
            self.accumulator = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        self.accumulator = cv2.addWeighted(self.cumulative, 0.9, img, 0.1, 0)

    def clear(self):
        self.accumulator = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)

    def get(self):
        return self.accumulator

class RawHandler(Handler):
    TITLE = 'Camera Connection'
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
    N = 30
    first_N_images = []
    def __init__(self):
        self.images = [None] * NewPixelsHandler.N
        self.index = 0
        self.BRIGHTNESS_THRESHOLD = 230
        self.loading_img = None
        self.title = f'Average of First {NewPixelsHandler.N} Frames'
        self.avg = None


    def add(self, img):
        # Skip if index is -1
        if self.index == -1:
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
            self.get()
            # cv2.imshow(self.title, self.avg)


    def get(self):
        # Return the average if it has already been calculated
        if hasattr(self, 'avg') and self.avg is not None:
            return self.avg
        
        if len(self.images) < NewPixelsHandler.N:
            return None
        
        # Calculate the average
        self.avg = np.zeros_like(self.images[0])
        # Average only the non-None images
        N_effective = NewPixelsHandler.N - len([i for i in self.images if i is None])
        for i in range(NewPixelsHandler.N):
            # Add the image to the average with a weight of 1/N_effective
            if self.images[i] is not None:
                self.avg = cv2.addWeighted(self.avg, 1, self.images[i], 1 / N_effective, 0)

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
        TITLE = 'Difference from Original'
        #TITLE2 = 'Cumulative Difference'
        LOADING_IMAGE = np.ones(img.shape, np.uint8) * 128
        if not self.isReady():
            cv2.imshow(TITLE, LOADING_IMAGE)
        else:
            diff = ImageParse.differenceImage(img, self.get())
            diff = ImageParse.blurImage(diff, 20)
            diff = ImageParse.aboveThreshold(diff, 50)

            #self.cumulative = cv2.addWeighted(self.cumulative, 0.9, diff, 0.1, 0)

            # convert image to RBG
            # diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            # find objects in the image
            diff = ImageParse.find_objects(diff)
            cv2.imshow(TITLE, diff)
            #cv2.imshow(TITLE2, self.cumulative)


class DifferenceHandler(Handler):
    TITLE = 'Difference'
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
        raise NotImplementedError('DifferenceHandler does not support clear()')
    
    def get(self):
        return self.diff


class Rendering:
    @staticmethod
    def showMultipleFrames(imgs, titles=None, title=None):
        N = len(imgs)
        # find the optimal p,q such that p*q >= N and p-q is minimized
        p = int(np.ceil(np.sqrt(N)))
        q = int(np.ceil(N / p))
        # Create a grid of p x q subplots
        fig, axs = plt.subplots(p, q, figsize=(10, 7))
        # Remove axis from all subplots
        for ax in axs.flat:
            ax.axis('off')
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
        # Black and white image
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return img

    @staticmethod
    def differenceImage(img1, img2):
        # Calculate difference
        diff = cv2.absdiff(img1, img2)
        # Apply threshold
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return diff

    @staticmethod
    def blurImage(img, factor=5):
        if factor % 2 == 0:
            factor += 1
        return cv2.GaussianBlur(img, (factor, factor), 0)

    @staticmethod
    def aboveThreshold(img, threshold):
        return cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)[1]
    
    @staticmethod
    def find_objects(img):
        # TODO: refactor this function
        # img is the output of the differenceImage function,
        # regarding the difference between the current frame and the average of the first N frames
        params = cv2.SimpleBlobDetector_Params()
        params.maxThreshold = 255
        params.minThreshold = 10
        # Filter by color (white blobs)
        params.filterByColor = True
        params.blobColor = 255

        # Filter by area
        params.filterByArea = True
        params.minArea = 40  # Minimum area of blob
        params.maxArea = 10000  # Maximum area of blob
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        brightnesses = {}
        # print the centerpoints (x,y) of the blobs
        for keypoint in keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            radius = int(keypoint.size / 2)
            # Get the mean brightness of the blob
            # If a pixel is outside the image, it is ignored
            blob_pixels = img[max(0, y - radius):min(img.shape[0], y + radius), max(0, x - radius):min(img.shape[1], x + radius)]
            # blob_pixels = img[y - radius:y + radius, x - radius:x + radius]
            mean_brightness = np.mean(blob_pixels) if len(blob_pixels) > 0 else 0
            brightnesses[keypoint] = mean_brightness
            # format with 2 decimal points
            # print(f'({x:.2f}, {y:.2f}), r={radius:.2f}, Mean Brightness: {mean_brightness:.2f}')

        # convert img to RBG
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # draw the keypoints
        for keypoint in keypoints:
            # Draw a circle around the blob
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            radius = int(keypoint.size / 2)
            brightness = brightnesses[keypoint]
            color = (0, 255, 0) if brightness > 60 else (255, 0, 0)
            cv2.circle(img, (x, y), radius, color, 2)
            # draw an inner circle with the mean brightness
            cv2.circle(img, (x, y), radius, (0, int(brightness), 0), -1)
            # Draw a cross on the blob
            cross_size = 2
            cv2.line(img, (x - cross_size, y), (x + cross_size, y), color, 1)
            cv2.line(img, (x, y - cross_size), (x, y + cross_size), color, 1)

        return img


class IO:
    def detectCameras():
        # first try to connect to CAMERA_INDEX
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if cam.isOpened():
            print(f'Camera @ index {CAMERA_INDEX} is connected')
            cam.release()
            return
        # Otherwise, try to connect to all cameras
        imgs = [None] * MAX_CAMERAS
        for i in range(MAX_CAMERAS):
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                print(f'Camera @ index {i} is connected')
                # Get a frame from the camera
                ret_val, imgs[i] = cam.read()
                cam.release()
        # Show all images in matplotlib window
        Rendering.showMultipleFrames(imgs, [f'Camera {i}' for i in range(MAX_CAMERAS)], f'Failed to connect to camera #{CAMERA_INDEX}\nAll Available Cameras')
        plt.show()

    def saveImage(img, path):
        cv2.imwrite(path, img)


def main():
    global CAMERA_INDEX
    IO.detectCameras()
    cam = cv2.VideoCapture(CAMERA_INDEX)
    rawHandler = RawHandler()
    newPixelsHandler = NewPixelsHandler()
    differenceHandler = DifferenceHandler()
    contoursHandler = ContoursHandler()

    while True:
        ret_val, img = cam.read()

        if not ret_val:
            print('Camera @ index 1 not connected')
            CAMERA_INDEX = int(input('Enter the index of the camera you want to connect to: '))
            cam = cv2.VideoCapture(CAMERA_INDEX)
            ret_val, img = cam.read()
            if not ret_val:
                print('Failed to connect to camera')
                break

        # Downsize and grayscale the image to better simulate IR camera - Remove this when IR camera is connected
        img = cv2.resize(img, (0, 0), fx=.5, fy=.5)
        img = ImageParse.toGrayscale(img)

        for handler in [rawHandler, newPixelsHandler, differenceHandler, contoursHandler]:
            handler.add(img)
            handler.display(img)
        
        if cv2.waitKey(1) == 32: # Whitespace
            newPixelsHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()