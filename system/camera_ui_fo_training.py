import cv2
import os
from datetime import datetime
from image_processing import ImageParse
import undistortion

SAVE_FOLDER = 'system/test_images'

# Ensure the output directory exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

class Camera:
    def __init__(self, index):
        self.index = index
        self.cam = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret_val, self.img = self.cam.read()
        self.img = ImageParse.resize_proportionally(self.img, 0.5)
        if not ret_val:
            raise RuntimeError(f"Failed to connect to camera @ index {self.index}")

    def read(self):
        ret_val, self.img = self.cam.read()
        self.img = ImageParse.resize_proportionally(self.img, 0.5)
        self.img = ImageParse.toGrayscale(self.img)
        self.img = undistortion.undistort(self.img)
        self.img = cv2.rotate(self.img, cv2.ROTATE_180)
        return self.img

# Callback for mouse events
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_FOLDER, f"{timestamp}.png")
        cv2.imwrite(filename, img)
        print(f"✅ Image saved: {filename}")

if __name__ == "__main__":
    cam = Camera(0)
    cv2.namedWindow("Camera")

    while True:
        img = cam.read()
        cv2.setMouseCallback("Camera", on_mouse, img)
        cv2.imshow("Camera", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.cam.release()
    cv2.destroyAllWindows()