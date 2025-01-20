import numpy as np
from constants import IMG_HEIGHT, IMG_WIDTH
import cv2


INITIAL_BLURRING_KERNEL = (3, 3)

HIGH_CEP_INDEX = 0.9
LOW_CEP_INDEX = 0.5
# sample rate - in frames, not in seconds
SAMPLE_RATE = 24
FRAMES_FOR_INITIALISATION = 30
BRIGHTNESS_THRESHOLD = 240

class Targets:
    """
    abstraction of the decision algorithm. todo: add a logic for extracting a coordinate and for resetting the image after a shot.
    """

    first_images = []

    def __init__(self):
        self.frame_number = 0
        self.contours_heatmap = None
        self.changes_heatmap = None
        self.yolo_centers = None
        self.contours_centers = None
        self.changes_centers = None
        self.target_queue = []

    def add(self, frame_number, contours_heatmap, changes_heatmap, yolo_centers):
        self.frame_number = frame_number
        self.contours_heatmap = contours_heatmap
        self.changes_heatmap = changes_heatmap
        # at the "initial frame", add all current objects to the queue
        if self.frame_number == SAMPLE_RATE//2:
            _, _, self.contours_centers = get_targets(contours_heatmap)
        if self.frame_number%SAMPLE_RATE == SAMPLE_RATE-1:
            self.yolo_centers = yolo_centers
            _, _, self.changes_centers = get_targets(changes_heatmap)
        # for centers in [self.changes_centers, self.contours_centers]:

    # def pop

def average_of_heatmaps(changes_map, contours_map):
    """Intersect two heatmaps

    Args:
        heatmap1 (np.ndarray): The first heatmap
        heatmap2 (np.ndarray): The second heatmap

    Returns:
        np.ndarray: The intersection of the two heatmaps
    """
    
    if (isinstance(changes_map, np.ndarray) and changes_map.size > 1) and (
        isinstance(contours_map, np.ndarray) and contours_map.size > 1
    ):
        if np.mean(changes_map) == 0:
            return contours_map
        if np.mean(contours_map) == 0:
            return changes_map
        # result = changes_map + contours_map
        #result = np.clip(result, 0, 255)
        result = changes_map
        return result
    if isinstance(changes_map, np.ndarray) and changes_map.size > 1:
        return changes_map
    if isinstance(contours_map, np.ndarray) and contours_map.size > 1:
        return contours_map
    return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)


def show_targets(title, image, targets):
    # If the image is empty, return [], [] and []
    if image is None:
        return [], [], []
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    circles_high, circles_low, centers = targets
    img = image.copy()
    LOW_COLOR = (0, 255, 0)
    HIGH_COLOR = (0, 0, 255)
    CENTER_COLOR = (255, 0, 0)
    for circle in circles_low:
        cv2.circle( 
            img,
            (int(circle[0][0]), int(circle[0][1])),
            int(circle[1]),
            LOW_COLOR,
            1,
        )
    for circle in circles_high:
        cv2.circle(
            img,
            (int(circle[0][0]), int(circle[0][1])),
            int(circle[1]),
            HIGH_COLOR,
            1,
        )
    for center in centers:
        cv2.circle(img, center, radius=1, color=CENTER_COLOR, thickness=-1)
    cv2.imshow(title, img)
    return circles_high, circles_low, centers


def get_targets(heatmap: cv2.typing.MatLike):
    """Generate targets from a heatmap.

    Args:
        heat_map (cv2.typing.MatLike): The heatmap to generate targets from

    Returns:
        Tuple: A tuple containing the targets for CEP_HIGH and CEP_LOW
    """
    high_intensity = int(HIGH_CEP_INDEX * 255)
    low_intensity = int(LOW_CEP_INDEX * 255)
    _, reduction_high = cv2.threshold(
        heatmap, high_intensity - 1, high_intensity, cv2.THRESH_BINARY
    )
    _, reduction_low = cv2.threshold(
        heatmap, low_intensity - 1, low_intensity, cv2.THRESH_BINARY
    )
    reduction_high = cv2.GaussianBlur(reduction_high, INITIAL_BLURRING_KERNEL, 0)
    reduction_low = cv2.GaussianBlur(reduction_low, INITIAL_BLURRING_KERNEL, 0)
    CEP_HIGH = cv2.Canny(reduction_high, 100, 150)
    CEP_LOW = cv2.Canny(reduction_low, 127, 128)
    contours_high, _ = cv2.findContours(
        CEP_HIGH, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    contours_low, _ = cv2.findContours(
        CEP_LOW, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    high_targets = []
    low_targets = []
    contour_centers = []
    for contour in contours_high:
        # add accurate CEP to list
        (x, y), radius = cv2.minEnclosingCircle(contour)
        new_circle = (x, y), radius
        high_targets.append(new_circle)
        # add contour center to list
        M = cv2.moments(contour)
        if not M["m00"] == 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            contour_centers.append((int(cx), int(cy)))
        else:
            contour_centers.append((int(x), int(y)))
    for contour in contours_low:
        # add inaccurate CEP to list
        (x, y), radius = cv2.minEnclosingCircle(contour)
        new_circle = (x, y), radius
        low_targets.append(new_circle)
    return high_targets, low_targets, contour_centers

