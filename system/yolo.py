from ultralytics import YOLO
import cv2
import numpy as np
from image_processing import Handler, ImageParse

class YOLOHandler(Handler):
    def __init__(self, model_path: str = 'yolov8n.pt'):
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

        return black_image


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