# Get all connected devices

import cv2
import numpy as np
import matplotlib.pyplot as plt

CAMERA_INDEX = 1

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
    fig, axs = plt.subplots(2, 5)
    for i in range(10):
        ax = axs[i // 5, i % 5]
        if imgs[i] is not None:
            ax.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Camera {i}')
    plt.show()


def show_webcam():
    detectCameras()
    cam = cv2.VideoCapture(CAMERA_INDEX)
    WINDOW_NAME = 'Camera Connection'
    prev = None
    while True:
        ret_val, img = cam.read()

        assert ret_val, 'Camera @ index 1 not connected'

        # Downsize the image to better simulate IR camera
        factor = .5
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor)

        processed_image = processImage(img)
        showEdges(img)

        if prev is not None:
            diff = differenceImage(prev, img)
            cv2.imshow('Difference', diff)
        prev = img

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