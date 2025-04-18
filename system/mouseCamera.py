import cv2
from image_processing import Handler


class MouseCameraHandler(Handler):
    TITLE = "Camera View"

    def __init__(self):
        super().__init__()
        self.mouseX = 0
        self.mouseY = 0
        self.img = None

    def add(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.addMouse()

    def get(self):
        return self.img

    def display(self):
        if self.img is None:
            return
        from fit import display_grid
        img = display_grid(self.img, False)
        text = "Press Space to shoot"
        textcolor = (130, 255, 0)
        cv2.putText(img, text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 1)
        cv2.imshow(MouseCameraHandler.TITLE, img)

    def mouse_callback(self, event, x, y, flags, param):
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