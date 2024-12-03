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


def process_video_area(camera_id=1, mask=None):
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
        edges = cv2.Canny(gray, 100, 200)

        # Mask edges to the user-defined area
        masked_edges = cv2.bitwise_and(edges, mask)

        # Find contours of the edges
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detected objects and calculate centers
        output_frame = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Ignore small objects
                cv2.drawContours(output_frame, [contour], -1, (0, 0, 255), 2)  # Red edges
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
    mask = initialize_camera()
    process_video_area(1, mask)

