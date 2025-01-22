import numpy as np
from constants import IMG_HEIGHT, IMG_WIDTH
import cv2
from contours import ContoursHandler
from changes import ChangesHandler
from bisect import bisect
from heapq import merge
from yolo import YOLOHandler


INITIAL_BLURRING_KERNEL = (3, 3)

HIGH_CEP_INDEX = 0.9
LOW_CEP_INDEX = 0.5
# sample rate - in frames, not in seconds
SAMPLE_RATE = 48
FRAMES_FOR_INITIALISATION = 15
BRIGHTNESS_THRESHOLD = 240


def insert_sorted(sorted_list, new_tuple):
    pos = bisect(sorted_list, new_tuple)
    sorted_list.insert(pos, new_tuple)
    return sorted_list


def insert_many_sorted_fast(sorted_list, new_tuples):
    sorted_list.extend(new_tuples)
    sorted_list.sort()
    return sorted_list


class Targets:
    """
    abstraction of the decision algorithm. todo: add a logic for extracting a coordinate and for resetting the image after a shot.
    """

    first_images = []

    def __init__(self):
        self.frame_number = 0
        self.contours_handler = ContoursHandler()
        self.changes_handler = ChangesHandler()
        self.yolo_handler = YOLOHandler() 
        self.img_changes = None
        self.yolo_centers = None
        self.contours_centers = None
        self.changes_centers = None
        self.low_targets = []
        self.high_targets = []
        self.target_queue = []

    def add(self, frame_number, img):
        self.frame_number = frame_number
        self.changes_handler.add(img)
        self.yolo_handler.add(img)
        self.yolo_centers = self.yolo_handler.get_centers()
        detected = self.yolo_handler.get()
        if isinstance(detected, np.ndarray) and detected.size > 1:
            cv2.imshow("yolo image", detected)
        self.contours_handler.add(img)
        self.contours_handler.display()
        self.img_contours = self.contours_handler.get()
        # at the "initial frame", add all current objects to the queue using contour identification
        # TODO: figure out how to use the contours properly
        # if self.frame_number == SAMPLE_RATE//4:
        #     print("pulling targets from contours")
        #     targets_contours = self.high_targets, self.low_targets, self.contours_centers = get_targets(self.img_contours)
        #     print("targets contours", self.contours_centers)
        #     show_targets("targets from contours", self.img_contours, targets_contours)
        #     # self.contours_centers = sorted(self.contours_centers, key=lambda x: x[0], reverse=True)
        #     # self.target_queue.extend(self.contours_centers)
        #     if len(self.contours_centers) > 0:
        #         # Remove from centers_contours any targets that are less than 20 pixels apart (unique targets)
        #         targets = []
        #         pixel_distance = 30
        #         for center in self.contours_centers:
        #             if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
        #                 insert_sorted(targets, center)
        #         self.target_queue = list(merge(self.target_queue, targets.copy()))

        if self.frame_number == SAMPLE_RATE//4:
            if len(self.yolo_centers) > 0:
                # Remove from centers_contours any targets that are less than 20 pixels apart (unique targets)
                targets = []
                pixel_distance = 30
                for center in self.yolo_centers:
                    if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
                        insert_sorted(targets, center)
                self.target_queue = list(merge(self.target_queue, targets.copy()))


        # at a constant rate SAMPLE_RATE, get all new objects in the image
        if self.frame_number%SAMPLE_RATE == SAMPLE_RATE-1 and isinstance(self.img_changes, np.ndarray) and self.img_changes.size > 1:
            print("pulling targets from changes")
            targets_changes = self.high_targets, self.low_targets, self.changes_centers = get_targets(self.img_changes)
            # show_targets("targets from changes", self.img_changes, targets_changes)
            # add the targets from the changes to the queue
            if len(self.changes_centers) > 0:
                # Remove from centers_changes any targets that are less than 30 pixels apart (unique targets)
                targets = []
                pixel_distance = 30
                for center in self.changes_centers:
                    if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
                        insert_sorted(targets, center)
                self.target_queue = list(merge(self.target_queue, targets.copy()))
            # reset the changes heatmap, so we get no duplicates
            # print("Queue:", self.target_queue)
            self.changes_handler.clear() 
            
    def pop(self):
        if self.target_queue:
            target = self.target_queue.pop(0)
            return target
        return
    
    def clear(self):
        self.changes_handler.clear() 


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

