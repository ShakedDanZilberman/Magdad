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
