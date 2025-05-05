from image_processing import Handler, ImageParse
import os
import numpy as np
import cv2
from ultralytics import YOLO
from openvino.runtime import Core
import time


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image to new_shape (h, w), preserving aspect ratio
    shape = img.shape[:2]  # current shape (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding into two sides
    dh /= 2

    # resize
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)


class YOLOHandler:
    def __init__(self,
                 model_path: str = 'last.pt',
                 imgsz: int = 640,
                 conf_thres: float = 0.5,
                 use_openvino: bool = True,
                 device: str = 'CPU'):
        # Resolve paths
        base = os.path.dirname(os.path.abspath(__file__))
        self.conf_threshold = conf_thres
        self.img = None
        self.bounding_boxes = []
        self.use_openvino = use_openvino
        self.device = device

        if not use_openvino:
            # Standard YOLO model
            self.imgsz = imgsz
            self.model = YOLO(os.path.join(base, model_path))
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            _ = self.model(dummy, imgsz=imgsz)
        else:
            # Load OpenVINO IR
            model_name = os.path.splitext(model_path)[0]
            export_dir = os.path.join(base, f"{model_name}_openvino_model")
            xml_path = os.path.join(export_dir, f"{model_name}.xml")
            bin_path = xml_path.replace('.xml', '.bin')
            if not (os.path.isfile(xml_path) and os.path.isfile(bin_path)):
                yolo = YOLO(os.path.join(base, model_path))
                yolo.export(format='openvino', dynamic=False, opset=11)

            ie = Core()
            model_ir = ie.read_model(model=xml_path)
            compiled = ie.compile_model(model=model_ir, device_name=device)
            self.compiled_model = compiled
            self.input_key = compiled.input(0)
            self.output_key = compiled.output(0)
            _, _, h, w = self.input_key.shape
            self.imgsz = (w, h)

    def add(self, img: np.ndarray):
        start_time = time.time()
        if img is None:
            return
        self.img = img.copy()
        self.bounding_boxes = []

        # Letterbox padding for both backends
        img_padded, scale, (dw, dh) = letterbox(img, new_shape=self.imgsz)
        # RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        inp = img_norm.transpose(2, 0, 1)[None, ...]  # NCHW

        if not self.use_openvino:
            # native YOLO inference uses its own letterbox internally
            results = self.model(img, imgsz=self.imgsz[0], conf=self.conf_threshold)
            res = results[0]
            for *xyxy, conf, cls in res.boxes.data.cpu().numpy():
                if conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                self.bounding_boxes.append({
                    'class': self.model.names[int(cls)],
                    'confidence': float(conf),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
        else:
            # OpenVINO inference
            out = self.compiled_model({self.input_key: inp})[self.output_key]
            for det in out[0]:
                vals = det.tolist()
                x1, y1, x2, y2, score, cls_id = vals[:6]
                if score < self.conf_threshold:
                    continue
                # reverse letterbox transform
                x1 = (x1 - dw) / scale
                y1 = (y1 - dh) / scale
                x2 = (x2 - dw) / scale
                y2 = (y2 - dh) / scale
                # clip and convert
                h0, w0 = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0, x2), min(h0, y2)
                self.bounding_boxes.append({
                    'class': str(int(cls_id)),
                    'confidence': float(score),
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)
                })

        elapsed = time.time() - start_time
        print(f"Frame processed in {elapsed*1000:.1f} ms")

    def get(self) -> np.ndarray:
        if self.img is None:
            return None
        mask = np.zeros_like(self.img)
        for b in self.bounding_boxes:
            cv2.rectangle(mask, (b['x1'], b['y1']), (b['x2'], b['y2']), 255, -1)
        return mask

    def get_centers(self):
        return [((b['x1'] + b['x2']) // 2, (b['y1'] + b['y2']) // 2)
                for b in self.bounding_boxes]

    def display(self):
        if self.img is None:
            return
        vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for b in self.bounding_boxes:
            color = (0, int(b['confidence'] * 255), 0)
            cv2.rectangle(vis, (b['x1'], b['y1']), (b['x2'], b['y2']), color, 2)
            label = f"{b['class']}:{b['confidence']:.2f}"
            cv2.putText(vis, label, (b['x1'], b['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow("YOLOv8 Detections", vis)
        cv2.imshow("BBox Mask", self.get())

    def clear(self):
        self.img = None
        self.bounding_boxes = []


if __name__ == "__main__":
    from cameraIO import detectCameras, Camera
    from mouseCamera import MouseCameraHandler
    from constants import CAMERA_INDEX_0

    detectCameras()
    cam = Camera(CAMERA_INDEX_0)
    handler = MouseCameraHandler()
    yoloHandler = YOLOHandler()

    cv2.namedWindow(handler.TITLE)
    cv2.setMouseCallback(handler.TITLE, handler.mouse_callback)
    frame_num = 0
    while True:
        img = cam.read()
        handler.add(img)
        handler.display()
        if frame_num%7==5:
            yoloHandler.add(img)
            yoloHandler.display()
        frame_num+=1
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
