from ultralytics import YOLO
import cv2
import numpy as np
from image_processing import Handler, ImageParse
import os

class YOLOHandler(Handler):
    def __init__(self, model_path: str = 'last.pt'):
        # get absolute path to current folder
        current_folder = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_folder, model_path)

        self.model = YOLO(model_path)
        self.img = None
        self.bounding_boxes = []
        self.conf_threshold = 0.5

    
    def add(self, img):
        """
        Add an image to the handler.

        Args:
            img (np.ndarray): The image to add to the handler.

        Returns:
            None
        """
        self.img = img
        # convert the image to RGB from grayscale
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # remove printing
        print("before prediction")
        results = self.model.predict(img, imgsz=640, stream=True, verbose=False)
        # convert the image back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.bounding_boxes = []
        for result in results:
            for box in result.boxes.data:  # Each box contains (x1, y1, x2, y2, confidence, class_id)
                x1, y1, x2, y2, confidence, class_id = box.cpu().numpy()
                if confidence >= self.conf_threshold:
                    self.bounding_boxes.append({
                        'class': self.model.names[int(class_id)],   # Class name
                        'confidence': float(confidence),            # Confidence score
                        'x1': int(x1),                              # Top-left corner x-coordinate
                        'y1': int(y1),                              # Top-left corner y-coordinate
                        'x2': int(x2),                              # Bottom-right corner x-coordinate
                        'y2': int(y2)                               # Bottom-right corner y-coordinate
                    })
        print("bounding boxes = ", self.bounding_boxes)


    def get(self):
        """
        Get the image from the handler.

        Returns:
            np.ndarray: The image from the handler.
        """
        if self.img is None:
            return None
        # Create a black image of the same size as the input image
        black_image = np.zeros_like(self.img)
        # Draw bounding boxes on the black image as white rectangles
        white = 255
        for box in self.bounding_boxes:
            topleft = (box['x1'], box['y1'])
            bottomright = (box['x2'], box['y2'])
            cv2.rectangle(black_image, topleft, bottomright, white, -1)
        cv2.imshow("image from get", black_image)
        return black_image
    

    def get_centers(self):
        """
        Get the centers of the bounding boxes.

        Returns:
            list[tuple]: The centers of the bounding boxes.
        """

        centers = []
        for box in self.bounding_boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            centers.append(center)
        return centers



    def display(self):
        """
        Display the image stored in the handler.
        Uses cv2.imshow() to display the image.

        Returns:
            None
        """
        if self.img is None:
            return
        image = self.img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        black = (0, 0, 0)
        for box in self.bounding_boxes:
            topleft = (box['x1'], box['y1'])
            bottomright = (box['x2'], box['y2'])
            certainty = int(box['confidence'] * 255)
            color = (0, certainty, 0)
            rectangle_thickness = 3
            cv2.rectangle(image, topleft, bottomright, color, rectangle_thickness)
            label = f"{box['class']} ({box['confidence']:.2f})"
            topleft_text = (box['x1'], box['y1'] + 12)
            # draw a filled rectangle for the label
            cv2.putText(image, label, topleft_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)
            

        cv2.imshow("YOLO Detections", image)
        # also show the black image with bounding boxes
        cv2.imshow("YOLO Bounding Boxes", self.get())

    def clear(self):
        """
        Clear the image and memory stored in the handler.
        """
        self.img = None
        self.bounding_boxes = []


if __name__ == "__main__":
    # read the images from camera of the computer and display the bounding boxes using the YOLOHandler class inside some loop
    global CAMERA_INDEX, timestep, laser_targets
    import fit
    from cameraIO import detectCameras
    from cameraIO import Camera
    from mouseCamera import MouseCameraHandler
    from main import CAMERA_INDEX

    detectCameras()
    cam = Camera(CAMERA_INDEX)
    handler = MouseCameraHandler()
    yoloHandler = YOLOHandler()
    # laser = threading.Thread(target=laser_thread)
    # laser.start()  # comment this line to disable the laser pointer

    cv2.namedWindow(handler.TITLE)
    cv2.setMouseCallback(handler.TITLE, handler.mouse_callback)

    while True:
        img = cam.read()

        handler.add(img)
        handler.display()

        yoloHandler.add(img)
        yoloHandler.display()


        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()