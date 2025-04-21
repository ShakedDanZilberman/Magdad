import numpy as np
from constants import IMG_HEIGHT, IMG_WIDTH
import cv2
from contours import ContoursHandler
from changes import ChangesHandler
from bisect import bisect
from heapq import merge
from yolo import YOLOHandler
from constants import INITIAL_BLURRING_KERNEL, HIGH_CEP_INDEX, LOW_CEP_INDEX, SAMPLE_RATE, FRAMES_FOR_INITIALISATION, BRIGHTNESS_THRESHOLD, MINIMAL_OBJECT_AREA

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
        self.frames_remaining_to_initialize = FRAMES_FOR_INITIALISATION

    def add(self, frame_number, img):
        self.frame_number = frame_number
        # print("frame number is", frame_number)
        # if self.frames_remaining_to_initialize > 0:
        self.changes_handler.add(img)
            # self.frames_remaining_to_initialize -= 1
        # print(self.frames_remaining_to_initialize, "frames remaining")
        # self.contours_handler.add(img)
        self.changes_handler.display()
        # self.img_contours = self.contours_handler.get()
        self.img_changes = self.changes_handler.get()

        # At the SAMPLE_RATE//4th frame, pull all targets from yolo detection
        if self.frame_number == 5: # SAMPLE_RATE//4:
            self.add_initial_targets_using_yolo(img)
            print("target queue: ", self.target_queue)

        # FIXME: isinstance(self.img_changes, np.ndarray) is always false, so no new targets are ever pulled from the changes image 
        # at a constant rate SAMPLE_RATE, get all new objects in the image
        if self.frame_number%SAMPLE_RATE == SAMPLE_RATE//2: 
            self.changes_handler.add(img)
            self.changes_handler.display()
            print(isinstance(self.img_changes, np.ndarray))
            if isinstance(self.img_changes, np.ndarray) and self.img_changes.size > 1:
                print("looking for changes")
                self.add_new_targets_to_queue()
                print("target queue: ", self.target_queue)


    def show_yolo_detection(self, img):
        '''shows what the AI model detected'''
        self.yolo_handler.add(img)
        self.yolo_centers = self.yolo_handler.get_centers()
        detected = self.yolo_handler.get()
        if isinstance(detected, np.ndarray) and detected.size > 1:
            cv2.imshow("yolo image", detected)


    def add_new_targets_to_queue(self):
        print("pulling targets from changes")
        targets_from_changes = _, _, self.changes_centers = get_targets_improved(self.img_changes)
        print("added new targets from changes:", self.changes_centers)
        show_targets("targets from changes", self.img_changes, targets_from_changes)
        # add the targets from the changes to the queue
        print("length of changes centers:", len(self.changes_centers))
        if len(self.changes_centers) > 0:
            # Remove from centers_changes any targets that are less than 30 pixels apart (unique targets)
            targets = []
            pixel_distance = 5
            for center in self.changes_centers:
                if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
                    insert_sorted(targets, center)
            self.target_queue = list(merge(self.target_queue, targets.copy()))
            # reset the changes heatmap, so we get no duplicates
        self.changes_handler.clear()
        self.frames_remaining_to_initialize = FRAMES_FOR_INITIALISATION 
        


    def add_initial_targets_using_contours(self, img):
        print("pulling targets from contours")
        targets_contours = self.high_targets, self.low_targets, self.contours_centers = get_targets(self.img_contours)
        print("targets contours", self.contours_centers)
        show_targets("targets from contours", self.img_contours, targets_contours)
        # self.contours_centers = sorted(self.contours_centers, key=lambda x: x[0], reverse=True)
        # self.target_queue.extend(self.contours_centers)
        if len(self.contours_centers) > 0:
            # Remove from centers_contours any targets that are less than 20 pixels apart (unique targets)
            targets = []
            pixel_distance = 30
            for center in self.contours_centers:
                if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
                    insert_sorted(targets, center)
            self.target_queue = list(merge(self.target_queue, targets.copy()))


    def add_initial_targets_using_yolo(self, img):
        self.show_yolo_detection(img)
        if len(self.yolo_centers) > 0:
            # Remove from centers_contours any targets that are less than 20 pixels apart (unique targets)
            targets = []
            pixel_distance = 30
            for center in self.yolo_centers:
                if all(np.linalg.norm(np.array(center) - np.array(target)) > pixel_distance for target in targets):
                    insert_sorted(targets, center)
            # add new targets to the target queue, sorted by x coordinate
            self.target_queue = list(merge(self.target_queue, targets.copy()))
            print("added targets from yolo")


    def pop(self):
        if self.target_queue:
            target = self.target_queue.pop(0)
            return target
        return
    
    def pop_closest_to_current_location(self, current_location):
        if self.target_queue:
            target_to_pop = min(self.target_queue, key=lambda target: abs(target[0] - current_location[0]))
            print("target to pop is: ", target_to_pop)
            self.target_queue.remove(target_to_pop)
            return target_to_pop, True
        return current_location, False


    def clear(self):
        print("clearing changes")
        self.changes_handler.clear() 


class GlobalTargets:
    """this class holds the targets from all the cameras.
    it saves a list of targets in the global coordinates system.
    it has a method to add targets from all the cameras, that uses homography to transform the targets to the global coordinates system.
    it has a method to get the targets in the global coordinates system.
    
    """
    def __init__(self, target_manager1, target_manager2, target_manager3):
        self.target_managers = [target_manager1, target_manager2, target_manager3]
        self.target_queue = []
        self.homography_matrixs = []

    def add_homography_matrix(self, homography_matrix):
        """add a homography matrix to the list of homography matrixs"""
        self.homography_matrixs.append(homography_matrix)

    def add(self, target_manager, camera_index):
        """add a target manager to the list of target managers"""
        # get the homography matrix for the camera index
        homography_matrix = self.homography_matrixs[camera_index]
        # get the targets from the target manager
        targets = target_manager.target_queue
        # transform the targets to the global coordinates system using the homography matrix
        targets = cv2.perspectiveTransform(np.array(targets), homography_matrix)
        # add the targets to the global target queue
        self.target_queue.extend(targets)
        # TODO: (ayala) remove duplicates from the target queue
        # TODO: (ayala) remove targets that are too close to each other
        # TODO: (ayala) test this method

    def pop_closest_to_current_location():
        # TODO: (ayala) implement this method
        pass



        





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


def old_show_targets(title, image, targets):
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


def show_targets(title, image, targets):
    """
    Draws low-confidence (green), high-confidence (red) circles and centers (blue) on the image.
    """
    if image is None:
        return [], [], []

    # Convert to color image only once
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    circles_high, circles_low, centers = targets

    LOW_COLOR = (0, 255, 0)
    HIGH_COLOR = (0, 0, 255)
    CENTER_COLOR = (255, 0, 0)
    THICKNESS = 1
    CENTER_RADIUS = 1

    # Draw low-confidence targets (green)
    for ((x, y), r) in circles_low:
        cv2.circle(img, (int(x), int(y)), int(r), LOW_COLOR, THICKNESS)

    # Draw high-confidence targets (red)
    for ((x, y), r) in circles_high:
        cv2.circle(img, (int(x), int(y)), int(r), HIGH_COLOR, THICKNESS)

    # Draw center points (blue)
    for (cx, cy) in centers:
        cv2.circle(img, (int(cx), int(cy)), CENTER_RADIUS, CENTER_COLOR, thickness=-1)

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
        # TODO: right now we ignore contours that are "too small" to avoid excess contours - 
        # possibly look into ignoring contours whose edges are too close
        if cv2.contourArea(contour) > MINIMAL_OBJECT_AREA:
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



def get_targets_improved(heatmap: cv2.typing.MatLike, scale: float = 0.5):
    """
    Generate targets from a heatmap using downscaling for performance.

    Args:
        heatmap (cv2.typing.MatLike): Original heatmap.
        scale (float): Factor to downscale the heatmap for faster processing.

    Returns:
        Tuple: Lists of high-confidence circles, low-confidence circles, and centers, all in original resolution.
    """
    if not isinstance(heatmap, np.ndarray) or heatmap.size <= 1:
        return [], [], []

    # Downscale heatmap
    heatmap_small = cv2.resize(heatmap, (0, 0), fx=scale, fy=scale)

    # Thresholding
    high_intensity = int(HIGH_CEP_INDEX * 255)
    low_intensity = int(LOW_CEP_INDEX * 255)
    _, reduction_high = cv2.threshold(heatmap_small, high_intensity - 1, 255, cv2.THRESH_BINARY)
    _, reduction_low = cv2.threshold(heatmap_small, low_intensity - 1, 255, cv2.THRESH_BINARY)

    # Blurring to reduce noise
    reduction_high = cv2.GaussianBlur(reduction_high, INITIAL_BLURRING_KERNEL, 0)
    reduction_low = cv2.GaussianBlur(reduction_low, INITIAL_BLURRING_KERNEL, 0)

    # Edge detection
    CEP_HIGH = cv2.Canny(reduction_high, 100, 150)
    CEP_LOW = cv2.Canny(reduction_low, 127, 128)

    # Contour detection
    contours_high, _ = cv2.findContours(CEP_HIGH, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_low, _ = cv2.findContours(CEP_LOW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    high_targets = []
    low_targets = []
    contour_centers = []

    for contour in contours_high:
        if cv2.contourArea(contour) > MINIMAL_OBJECT_AREA * (scale ** 2):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            high_targets.append(((x / scale, y / scale), radius / scale))

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x, y
            contour_centers.append((cx / scale, cy / scale))

    for contour in contours_low:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        low_targets.append(((x / scale, y / scale), radius / scale))

    # Cast final results to int for consistency
    high_targets = [((int(x), int(y)), int(r)) for ((x, y), r) in high_targets]
    low_targets = [((int(x), int(y)), int(r)) for ((x, y), r) in low_targets]
    contour_centers = [(int(x), int(y)) for (x, y) in contour_centers]

    return high_targets, low_targets, contour_centers
