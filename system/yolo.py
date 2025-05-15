from image_processing import Handler, ImageParse
import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
# from openvino import Core
from ultralytics.data.augment import LetterBox
from constants import *

class YOLOHandler:
    def __init__(self, model_path: str = 'best_new_training_openvino_model', imgsz: int = 320, conf_thres: float = 0.5):
        # Resolve model path
        base = os.path.dirname(os.path.abspath(__file__))
        print("Model loaded")
        self.model = YOLO(os.path.join(base, model_path))
        # self.model.export(format='openvino')

        self.imgsz = imgsz
        self.conf_threshold = conf_thres
        self.img = None
        self.bounding_boxes = []
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz))
        # inside YOLOHandler.__init__:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = self.model(dummy, imgsz=imgsz)

    def add(self, img: np.ndarray):
        if img is None:
            return

        self.img = img.copy()
        ih, iw = img.shape[:2]

        # Compute scaling factor and padding
        scale = min(self.imgsz / iw, self.imgsz / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2

        # Resize and pad the image
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz), 114, dtype=np.uint8)
        canvas[dy:dy+nh, dx:dx+nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(rgb, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
        res = results[0]

        self.bounding_boxes = []

        if res.boxes.shape[0] == 0:
            return

        for *xyxy, conf, cls in res.boxes.data.cpu().numpy():
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(float, xyxy)

            # Rescale boxes to original image coordinates
            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale

            self.bounding_boxes.append({
                'class': self.model.names[int(cls)],
                'confidence': float(conf),
                'x1': int(max(0, min(iw, x1))),
                'y1': int(max(0, min(ih, y1))),
                'x2': int(max(0, min(iw, x2))),
                'y2': int(max(0, min(ih, y2)))
            })

    def get(self) -> np.ndarray:
        """Returns a black mask with white filled bboxes."""
        if self.img is None:
            return None
        mask = np.zeros_like(self.img)
        for b in self.bounding_boxes:
            cv2.rectangle(mask, (b['x1'],b['y1']), (b['x2'],b['y2']), 255, -1)
        return mask

    def get_centers(self):
        return [((b['x1']+b['x2'])//2, (b['y1']+b['y2'])//10)
                for b in self.bounding_boxes]

    def display(self):
        if self.img is None:
            return
        vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for b in self.bounding_boxes:
            color = (0, int(b['confidence']*255), 0)
            cv2.rectangle(vis, (b['x1'],b['y1']), (b['x2'],b['y2']), color, 2)
            label = f"{b['class']}:{b['confidence']:.2f}"
            cv2.putText(vis, label, (b['x1'],b['y1']-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow("YOLOv8 Detections", vis)
        # print(self.get_centers())
        # cv2.imshow("BBox Mask", self.get())
        # # press Escape to exit
        # if cv2.waitKey(1) == 27:
        #     sys.exit(0)

    def prepare_to_show(self):
        if self.img is None:
            return
        vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for b in self.bounding_boxes:
            color = (0, int(b['confidence']*255), 0)
            cv2.rectangle(vis, (b['x1'],b['y1']), (b['x2'],b['y2']), color, 2)
            label = f"{b['class']}:{b['confidence']:.2f}"
            cv2.putText(vis, label, (b['x1'],b['y1']-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        self.vis = vis
    
    def get_vis(self):
        if self.img is None:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        return self.vis

    def clear(self):
        self.img = None
        self.bounding_boxes = []



if __name__ == "__main__":
    # read the images from camera of the computer and display the bounding boxes using the YOLOHandler class inside some loop
    global timestep, laser_targets
    # import fit
    from cameraIO import detectCameras
    from cameraIO import Camera, ImageParse
    from mouseCamera import MouseCameraHandler
    from constants import CAMERA_INDEX_0
    import undistortion
    detectCameras()
    cam = Camera(CAMERA_INDEX_0)
    # handler = MouseCameraHandler()
    yoloHandler = YOLOHandler()
    # laser = threading.Thread(target=laser_thread)
    # laser.start()  # comment this line to disable the laser pointer

    # cv2.namedWindow(handler.TITLE)
    # cv2.setMouseCallback(handler.TITLE, handler.mouse_callback)

    while True:
        img = cam.read()
        # handler.add(img)
        # handler.display()

        yoloHandler.add(img)
        yoloHandler.display()


        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()




