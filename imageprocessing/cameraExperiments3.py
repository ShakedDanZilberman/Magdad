# Get all connected devices

import cv2
import numpy as np
import matplotlib.pyplot as plt

CAMERA_INDEX = 1
averageFirstNFrames = 30
WINDOW_NAME = 'Camera Connection'
image_index = 0
first_N_images = []

class FirstNImagesHandler:
    def __init__(self, N):
        self.N = N
        self.images = [None] * N
        self.index = 0
        self.BRIGHTNESS_THRESHOLD = 230

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
            cv2.imshow(f'Average of First {self.N} Frames', self.avg)

    def getAverage(self):
        # Return the average if it has already been calculated
        if hasattr(self, 'avg') and self.avg is not None:
            return self.avg
        
        if len(self.images) < self.N:
            return None
        
        # Calculate the average
        self.avg = np.zeros_like(self.images[0])
        for i in range(self.N):
            # Add the image to the average with a weight of 1/N
            self.avg = cv2.addWeighted(self.avg, 1, self.images[i], 1 / self.N, 0)
        return self.avg

    def clear(self):
        # Clear the images and the average, and reset the index
        self.images = [None] * self.N
        self.avg = None
        self.index = 0

    def isReady(self):
        return self.index == -1

def processImage(img):
    # Black and white image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def showEdges(img):
    # Detect Canny edges
    edges = cv2.Canny(img, 100, 200)
    # Detect contours
    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # Draw contours on a black image
    # black = np.zeros_like(img)
    # cv2.drawContours(black, contours, -1, (255, 255, 255), 1)
    # # Show
    # cv2.imshow('Contours', black)
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

def show10Images(imgs, titles=None):
    fig, axs = plt.subplots(2, 5)
    for i in range(10):
        ax = axs[i // 5, i % 5]
        ax.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])
    plt.show()

def detectCameras():
    # first try to connect to CAMERA_INDEX
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if cam.isOpened():
        print(f'Camera @ index {CAMERA_INDEX} is connected')
        cam.release()
        return
    # Otherwise, try to connect to all
    imgs = [None] * 10
    for i in range(10):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f'Camera @ index {i} is connected')
            # Get a frame from the camera
            ret_val, imgs[i] = cam.read()
            cam.release()
    # Show all images in matplotlib window
    show10Images(imgs, [f'Camera {i}' for i in range(10)])
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
        factor = .5
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor)

        processed_image = processImage(img)
        showEdges(img)

        firstFramesHandler.addImage(img)

        if prev is not None:
            diff = differenceImage(prev, img)
            cv2.imshow('Difference', diff)
        prev = img

        if firstFramesHandler.isReady():
            diff = differenceImage(img, firstFramesHandler.getAverage())
            cv2.imshow('Difference from Original', diff)


        cv2.imshow(WINDOW_NAME, processed_image)
        # Press Escape or close the window to exit
        if cv2.waitKey(1) == 27:
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    show_webcam()


if __name__ == '__main__':
    main()