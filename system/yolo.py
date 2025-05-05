from image_processing import Handler, ImageParse
import os
import numpy as np
import cv2
from ultralytics import YOLO
# from openvino.runtime/ import Core



class YOLOHandler:
    def __init__(self, model_path: str = 'last.pt', imgsz: int = 320, conf_thres: float = 0.5):
        # Resolve model path
        base = os.path.dirname(os.path.abspath(__file__))
        self.model = YOLO(os.path.join(base, model_path))
        # self.model.export('openvino')
        self.imgsz = imgsz
        self.conf_threshold = conf_thres
        self.img = None
        self.bounding_boxes = []
        # inside YOLOHandler.__init__:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = self.model(dummy, imgsz=imgsz)

    def add(self, img: np.ndarray):
        if img is None:
            return

        self.img = img.copy()

        # Resize and convert to RGB
        resized = cv2.resize(img, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(rgb, imgsz=self.imgsz, conf=self.conf_threshold)
        res = results[0]

        self.bounding_boxes = []

        if res.boxes.shape[0] == 0:
            return

        for *xyxy, conf, cls in res.boxes.data.cpu().numpy():
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            self.bounding_boxes.append({
                'class': self.model.names[int(cls)],
                'confidence': float(conf),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
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
        return [((b['x1']+b['x2'])//2, (b['y1']+b['y2'])//2)
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
        cv2.imshow("BBox Mask", self.get())

    def clear(self):
        self.img = None
        self.bounding_boxes = []

# class YOLOHandler:
#     def __init__(self,
#                  model_path: str = 'last.pt',
#                  imgsz: int = 320,
#                  conf_thres: float = 0.5,
#                  use_openvino: bool = True,
#                  device: str = 'CPU'):
#         # Resolve model path
#         base = os.path.dirname(os.path.abspath(__file__))
#         self.imgsz = imgsz
#         self.conf_threshold = conf_thres
#         self.img = None
#         self.bounding_boxes = []
#         self.use_openvino = use_openvino
#         self.device = device

#         if not use_openvino:
#             # Standard YOLOv8 model
#             self.model = YOLO(os.path.join(base, model_path))
#             # Warm up
#             dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
#             _ = self.model(dummy, imgsz=imgsz)
#         else:
#             # Export to OpenVINO IR if not already done
#             xml = os.path.join(base, 'openvino_model', os.path.splitext(model_path)[0] + '.xml')
#             if not os.path.isfile(xml):
#                 # perform export
#                 yolo = YOLO(os.path.join(base, model_path))
#                 yolo.export(format='openvino', dynamic=False, opset=11)
#             # Load OpenVINO model
#             ie = Core()
#             model_ir = ie.read_model(model=xml)
#             self.compiled_model = ie.compile_model(model=model_ir, device_name=device)
#             self.input_key = self.compiled_model.input(0)
#             self.output_key = self.compiled_model.output(0)

#     def add(self, img: np.ndarray):
#         if img is None:
#             return
#         self.img = img.copy()
#         self.bounding_boxes = []

#         # Preprocess
#         resized = cv2.resize(img, (self.imgsz, self.imgsz))
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         inp = rgb.transpose(2, 0, 1)[None, ...]  # NCHW

#         if not self.use_openvino:
#             # YOLOv8 native inference
#             results = self.model(rgb, imgsz=self.imgsz, conf=self.conf_threshold)
#             res = results[0]
#             for *xyxy, conf, cls in res.boxes.data.cpu().numpy():
#                 if conf < self.conf_threshold:
#                     continue
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 self.bounding_boxes.append({
#                     'class': self.model.names[int(cls)],
#                     'confidence': float(conf),
#                     'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
#                 })
#         else:
#             # OpenVINO inference
#             out = self.compiled_model({self.input_key: inp})[self.output_key]
#             # out shape: [1, num_detections, 6] -> [x1, y1, x2, y2, score, class_id]
#             for det in out[0]:
#                 x1, y1, x2, y2, score, cls_id = det.tolist()
#                 if score < self.conf_threshold:
#                     continue
#                 # scale coords back to original image
#                 h, w = img.shape[:2]
#                 x_scale = w / self.imgsz
#                 y_scale = h / self.imgsz
#                 self.bounding_boxes.append({
#                     'class': str(int(cls_id)),
#                     'confidence': float(score),
#                     'x1': int(x1 * x_scale),
#                     'y1': int(y1 * y_scale),
#                     'x2': int(x2 * x_scale),
#                     'y2': int(y2 * y_scale)
#                 })

#     def get(self) -> np.ndarray:
#         """Returns a black mask with white filled bboxes."""
#         if self.img is None:
#             return None
#         mask = np.zeros_like(self.img)
#         for b in self.bounding_boxes:
#             cv2.rectangle(mask, (b['x1'], b['y1']), (b['x2'], b['y2']), 255, -1)
#         return mask

#     def get_centers(self):
#         return [((b['x1'] + b['x2']) // 2, (b['y1'] + b['y2']) // 2)
#                 for b in self.bounding_boxes]

#     def display(self):
#         if self.img is None:
#             return
#         vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
#         for b in self.bounding_boxes:
#             color = (0, int(b['confidence'] * 255), 0)
#             cv2.rectangle(vis, (b['x1'], b['y1']), (b['x2'], b['y2']), color, 2)
#             label = f"{b['class']}:{b['confidence']:.2f}"
#             cv2.putText(vis, label, (b['x1'], b['y1'] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#         cv2.imshow("YOLOv8 Detections", vis)
#         cv2.imshow("BBox Mask", self.get())

#     def clear(self):
#         self.img = None
#         self.bounding_boxes = []


if __name__ == "__main__":
    # read the images from camera of the computer and display the bounding boxes using the YOLOHandler class inside some loop
    global timestep, laser_targets
    #import fit
    from cameraIO import detectCameras
    from cameraIO import Camera
    from mouseCamera import MouseCameraHandler
    from constants import *

    detectCameras()
    cam = Camera(CAMERA_INDEX_0)
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