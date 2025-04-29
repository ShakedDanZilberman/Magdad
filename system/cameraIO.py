import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_processing import ImageParse
import undistortion
from constants import FPS

MAX_CAMERAS = 10

def detectCameras():
    """Detect all connected cameras and display their images in a grid using matplotlib

    If the camera at CAMERA_INDEX is connected, the function will print a message and return.
    """
    from constants import CAMERA_INDEX
    # first try to connect to CAMERA_INDEX
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if cam.isOpened():
        print(f"Camera @ index {CAMERA_INDEX} is connected")
        cam.release()
        return True
    # Otherwise, try to connect to all cameras
    showAllCameras(CAMERA_INDEX)
    return False


def showAllCameras(camera_index=-1):
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
        (f"Failed to connect to camera #{camera_index}\n" if camera_index > -1 else "") + "All Available Cameras",
    )
    plt.show()


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
        print("index is", self.index)
        self.cam = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.cam.set(cv2.CAP_PROP_FPS, 10)
        # actual_fps = self.cam.get(cv2.CAP_PROP_FPS)
        # print(f"Actual FPS set: {actual_fps}")
        ret_val, self.img = self.cam.read()
        self.img = ImageParse.resize_proportionally(self.img, 0.5)
        if not ret_val:
            print(f"Camera @ index {self.index} not connected")
            self.index = int(input("Enter the index of the camera you want to connect to: "))
            self.cam = cv2.VideoCapture(self.index)
            img = self.cam.read()
            if not ret_val:
                print("Failed to connect to camera")
                return

    def read(self):
        ret_val, self.img = self.cam.read()
        self.img = ImageParse.resize_proportionally(self.img, 0.5)
        self.img = ImageParse.toGrayscale(self.img)
        self.img = undistortion.undistort(self.img)
        return self.img
    

# if __name__ == "__main__":
#     # display image from camera index 1
#     cam = Camera(1)
#     while True:
#         img = cam.read()
#         cv2.imshow("Camera", img)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cam.cam.release()

if __name__ == "__main__":
    showAllCameras()