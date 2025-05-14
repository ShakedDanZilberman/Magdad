import argparse
import cv2
import numpy as np
from openvino.runtime import Core

def preprocess(image: np.ndarray, target_size=(640, 640)):
    """
    Resize with letterbox (padding) to keep aspect ratio,
    convert to blob (NCHW), and normalize [0,1].
    """
    ih, iw = image.shape[:2]
    h, w = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    # letterbox resize
    resized = cv2.resize(image, (nw, nh))
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    dx, dy = (w - nw) // 2, (h - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized

    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1/255.0, size=(w, h), swapRB=True)
    return blob, scale, dx, dy

def postprocess(orig_image: np.ndarray, outputs, scale, dx, dy, conf_thresh=0.3):
    """
    Parse raw outputs and draw boxes on orig_image.
    """
    ih, iw = orig_image.shape[:2]
    detections = outputs[0]  # shape: [N,6]
    for x, y, w, h, conf, cls in detections:
        if conf < conf_thresh:
            continue
        # Convert from letterbox coordinates back to original image coords
        x = (x - dx) / scale
        y = (y - dy) / scale
        w /= scale
        h /= scale

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(orig_image, f"{int(cls)}:{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def main(args):
    # 1. Load model
    ie = Core()
    model = ie.read_model(model=args.model_xml)                            # :contentReference[oaicite:0]{index=0}
    compiled = ie.compile_model(model=model, device_name="CPU")             # :contentReference[oaicite:1]{index=1}
    output_layer = compiled.output(0)

    # 2. Read image
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Image not found: {args.image}")

    # 3. Preprocess
    blob, scale, dx, dy = preprocess(img, target_size=(args.width, args.height))

    # 4. Inference
    outputs = compiled([blob])[output_layer]

    # 5. Postprocess & visualize
    postprocess(img, outputs, scale, dx, dy, conf_thresh=args.conf)
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-xml", type=str, required=True,
                   help="Path to OpenVINO .xml file")
    p.add_argument("--image", type=str, required=True,
                   help="Path to test image")
    p.add_argument("--width", type=int, default=640,
                   help="Model input width")
    p.add_argument("--height", type=int, default=640,
                   help="Model input height")
    p.add_argument("--conf", type=float, default=0.3,
                   help="Confidence threshold")
    args = p.parse_args()
    main(args)
