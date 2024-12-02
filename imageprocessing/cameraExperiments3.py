# Get all connected devices

import cv2
import numpy as np

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

def show_webcam():
    cam = cv2.VideoCapture(1)
    WINDOW_NAME = 'Camera Connection'
    while True:
        ret_val, img = cam.read()

        # Downsize the image to better simulate IR camera
        factor = 0.5
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor)

        processed_image = processImage(img)
        showEdges(img)

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