import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_processing import toGrayscale

MAX_CAMERAS = 10

def detectCameras():
    """Detect all connected cameras and display their images in a grid using matplotlib

    If the camera at CAMERA_INDEX is connected, the function will print a message and return.
    """
    from main import CAMERA_INDEX
    # first try to connect to CAMERA_INDEX
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if cam.isOpened():
        print(f"Camera @ index {CAMERA_INDEX} is connected")
        cam.release()
        return True
    # Otherwise, try to connect to all cameras
    imgs = [None] * MAX_CAMERAS
    for i in range(MAX_CAMERAS):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"Camera @ index {i} is connected")
            # Get a frame from the camera
            ret_val, imgs[i] = cam.read()
            cam.release()
    # Show all images in matplotlib window
    showMultipleFrames(
        imgs,
        [f"Camera {i}" for i in range(MAX_CAMERAS)],
        f"Failed to connect to camera #{CAMERA_INDEX}\nAll Available Cameras",
    )
    plt.show()
    return False


def showMultipleFrames(imgs, titles=None, title=None):
    """
    Display multiple images in a grid using matplotlib.

    Args:
        imgs (list[np.ndarray]): List of images to display
        titles (list[str], optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to None.
    """
    N = len(imgs)
    # find the optimal p,q such that p*q >= N and p-q is minimized
    p = int(np.ceil(np.sqrt(N)))
    q = int(np.ceil(N / p))

    # Create a grid of p x q subplots
    fig, axs = plt.subplots(p, q, figsize=(10, 7))

    # Remove axis from all subplots
    for ax in axs.flat:
        ax.axis("off")
    
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

class Camera:
    def __init__(self, index):
        self.index = index
        self.cam = cv2.VideoCapture(index)

        ret_val, self.img = self.cam.read()
        if not ret_val:
            print("Camera @ index 1 not connected")
            self.index = int(input("Enter the index of the camera you want to connect to: "))
            self.cam = cv2.VideoCapture(self.index)
            img = self.cam.read()
            if not ret_val:
                print("Failed to connect to camera")
                return

    def read(self):
        ret_val, self.img = self.cam.read()
        self.img = toGrayscale(self.img)
        return self.img