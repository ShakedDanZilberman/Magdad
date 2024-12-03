# Get all connected devices

import cv2
import numpy as np
import matplotlib.pyplot as plt

CAMERA_INDEX = 1
averageFirstNFrames = 30
WINDOW_NAME = 'Camera Connection'
MAX_CAMERAS = 10
image_index = 0
first_N_images = []

class FirstNImagesHandler:
    def __init__(self, N):
        self.N = N
        self.images = [None] * N
        self.index = 0
        self.BRIGHTNESS_THRESHOLD = 230
        self.loading_img = None
        self.title = f'Average of First {self.N} Frames'

    def addImage(self, img):
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
        if self.index == self.N:
            self.index = -1
            self.getAverage()
            # cv2.imshow(self.title, self.avg)

    def getAverage(self):
        # Return the average if it has already been calculated
        if hasattr(self, 'avg') and self.avg is not None:
            return self.avg
        
        if len(self.images) < self.N:
            return None
        
        # Calculate the average
        self.avg = np.zeros_like(self.images[0])
        # Average only the non-None images
        N_effective = self.N - len([i for i in self.images if i is None])
        for i in range(self.N):
            # Add the image to the average with a weight of 1/N_effective
            if self.images[i] is not None:
                self.avg = cv2.addWeighted(self.avg, 1, self.images[i], 1 / N_effective, 0)
        return self.avg

    def clear(self):
        # Clear the images and the average, and reset the index
        self.shape = self.images[0].shape
        self.images = [None] * self.N
        self.avg = None
        self.index = 0

        self.loading_img = np.ones(self.shape, np.uint8) * 128

        # Show the loading image
        cv2.imshow(self.title, self.loading_img)

    def isReady(self):
        return self.index == -1
    
    def displayDifference(self, img):
        if not self.isReady():
            cv2.imshow('Difference from Original', np.ones(img.shape, np.uint8) * 128)
        diff = differenceImage(img, self.getAverage())
        diff = blurImage(diff, 20)
        diff = aboveThreshold(diff, 30)
        cv2.imshow('Difference from Original', diff)

def processImage(img):
    # Black and white image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def showEdges(img):
    # Detect Canny edges
    edges = cv2.Canny(img, 100, 200)
    # Detect contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a black image
    black = np.zeros_like(img)
    cv2.drawContours(black, contours, -1, (255, 255, 255), 1)
    # Show
    cv2.imshow('Contours', black)
    cv2.imshow('Edges', edges)

def differenceImage(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Calculate difference
    diff = cv2.absdiff(gray1, gray2)
    # Threshold
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return diff

def showMultipleFrames(imgs, titles=None):
    N = len(imgs)
    # find the optimal p,q such that p*q >= N and p-q is minimized
    p = int(np.ceil(np.sqrt(N)))
    q = int(np.ceil(N / p))
    fig, axs = plt.subplots(p, q, figsize=(20, 10))
    for ax in axs.flat:
        ax.axis('off')
    for i in range(N):
        ax = axs[i // q, i % q]
        if imgs[i] is not None:
            ax.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        if titles is not None:
            ax.set_title(titles[i])
    plt.show()

def blurImage(img, factor=5):
    if factor % 2 == 0:
        factor += 1
    return cv2.GaussianBlur(img, (factor, factor), 0)

def aboveThreshold(img, threshold):
    return cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)[1]

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
    showMultipleFrames(imgs, [f'Camera {i}' for i in range(MAX_CAMERAS)])
    plt.show()


def show_webcam():
    detectCameras()
    cam = cv2.VideoCapture(CAMERA_INDEX)
    firstFramesHandler = FirstNImagesHandler(averageFirstNFrames)
    prev = None

    while True:
        ret_val, img = cam.read()

        assert ret_val, 'Camera @ index 1 not connected'

        # Downsize the image to better simulate IR camera
        img = cv2.resize(img, (0, 0), fx=.5, fy=.5)

        processed_image = processImage(img)
        showEdges(img)

        firstFramesHandler.addImage(img)

        if prev is not None:
            diff = differenceImage(prev, img)
            cv2.imshow('Difference', diff)
        prev = img

        firstFramesHandler.displayDifference(img)
        if cv2.waitKey(1) == 32: # Whitespace
            firstFramesHandler.clear()


        cv2.imshow(WINDOW_NAME, processed_image)
        # Press Escape or close the window to exit
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    show_webcam()


if __name__ == '__main__':
    main()