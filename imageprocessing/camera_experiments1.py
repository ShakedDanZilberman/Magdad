import cv2
import numpy as np

# Global variables for calibration
calibration_mode = True
points = []


def draw_polygon(event, x, y, flags, param):
    """Callback function to collect points for the table area."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def calculate_center(contour):
    """Calculate the center of a contour."""
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return None


def initialize_camera(camera_id=1):
    """initializing the camera by returning a mask of a relavent polygon of the image with objects to detect"""
    global calibration_mode, points

    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", draw_polygon)

    while calibration_mode:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display frame for calibration
        temp_frame = frame.copy()
        for i in range(1, len(points)):
            cv2.line(temp_frame, points[i - 1], points[i], (0, 255, 0), 2)
        cv2.imshow("Calibration", temp_frame)

        # Check if the window is closed
        if cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) < 1:
            print("Calibration window closed. Exiting...")
            calibration_mode = False
            break

        # Wait for user input
        key = cv2.waitKey(1)
        if key == ord('q') and len(points) >= 4:  # Press 'q' to finish calibration
            calibration_mode = False
            cv2.destroyWindow("Calibration")

    # Ensure points are defined and create a mask for the area
    mask = np.zeros_like(frame[:, :, 0])
    if len(points) >= 4:
        points_np = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_np], 255)

    cap.release()
    cv2.destroyAllWindows()

    return mask


def blob_detection(frame):
    """Detect blobs in a frame using the SimpleBlobDetector.
    blures the image and detects blobs in the image
    does not detect edges that are not closed shapes
    """
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(frame)

    # Draw detected blobs as red circles
    output_frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_frame

def process_video_area(camera_id=1):
    """Main function to capture video, detect objects, and display results."""
    cap = cv2.VideoCapture(camera_id)


    # Start the live video feed
    cv2.namedWindow("Object Detection")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Check if the window is closed
        if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Object Detection window closed. Exiting...")
            break

        # Convert frame to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 150, 200)


        #use diliation to fill in the edges
        kernel_small = np.ones((3, 3), np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        kernel_big = np.ones((7, 7), np.uint8)

        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        opening = cv2.morphologyEx(dilated_edges, cv2.MORPH_OPEN, kernel_big)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        erode = cv2.erode(closing, kernel_small, iterations=4)

        # dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # opening = cv2.morphologyEx(dilated_edges, cv2.MORPH_OPEN, kernel_big)
        # dilated_2 = cv2.dilate(opening, kernel, iterations=2)
        # closing = cv2.morphologyEx(dilated_2, cv2.MORPH_CLOSE, kernel)
        #
        # erode = cv2.erode(closing, kernel_small, iterations=4)

        cv2.imshow("Dilated Edges", erode)


        # Mask edges to the user-defined area
        #masked_edges = cv2.bitwise_and(erode, mask)

        # Find contours of the edges
        contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        # contours_poly = [None] * len(contours)
        # boundRect = [None] * len(contours)
        # for i, c in enumerate(contours):
        #     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        #     boundRect[i] = cv2.boundingRect(contours_poly[i])
        # for i in range(len(contours)):
        #     color = (0, 0, 255)
        #     cv2.drawContours(frame, contours_poly, i, color)
        #     cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #                   (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)




        # Draw detected objects and calculate centers
        output_frame = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Ignore small objects
                cv2.drawContours(output_frame, [contour], -1, (0, 0, 255), cv2.FILLED)  # Red edges
                center = calculate_center(contour)
                if center:
                    cv2.circle(output_frame, center, 5, (255, 0, 0), -1)  # Blue center point

        # Display the result
        cv2.imshow("Object Detection", output_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    process_video_area(1)

