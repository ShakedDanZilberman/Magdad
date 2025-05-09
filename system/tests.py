import cv2
import numpy as np

# Define mouse callback functions for each window
def mouse_callback_window1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Window 1: Left click at ({x}, {y})")

def mouse_callback_window2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Window 2: Left click at ({x}, {y})")

# Create two named windows
cv2.namedWindow('Window1')
cv2.namedWindow('Window2')

# Set mouse callbacks for each window
cv2.setMouseCallback('Window1', mouse_callback_window1)
cv2.setMouseCallback('Window2', mouse_callback_window2)

# Create blank images for display
image1 = np.zeros((300, 300, 3), dtype=np.uint8)
image2 = np.zeros((300, 300, 3), dtype=np.uint8)

while True:
    cv2.imshow('Window1', image1)
    cv2.imshow('Window2', image2)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cv2.destroyAllWindows()
