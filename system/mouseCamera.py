import cv2
from image_processing import Handler


class MouseCameraHandler(Handler):
    TITLE = "Camera View"

    def __init__(self):
        super().__init__()
        self.mouseX = 0
        self.mouseY = 0
        self.img = None
        self.clicks = []
        self._new_click = False

    def add(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.addMouse()

    def get(self):
        return self.img

    def display(self, index=0):
        if self.img is None:
            return

        cv2.imshow("camera " + str(index) + " view", self.img)

    def mouse_callback_2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouseX = x
            self.mouseY = y

    def clear(self):
        self.img = None

    def addMouse(self):
        """Add a mouse pointer to the image.
        The mouse pointer is a green circle with a radius of 3 pixels (hardcoded in this function).
        """
        r = 3
        color = (0, 255, 0)

        if self.img is None:
            return
        
        cv2.circle(self.img, self.getMousePosition(), r, color, -1)

    def getMousePosition(self) -> tuple:
        return self.mouseX, self.mouseY


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked position in pixels: ({x}, {y})")
            self.clicks.append((x, y))

    def has_new_clicks(self):
        return len(self.clicks) > 0

    def get_clicks(self):
        clicks = self.clicks.copy()
        self.clicks.clear()
        print("clicks", clicks)
        return clicks
    