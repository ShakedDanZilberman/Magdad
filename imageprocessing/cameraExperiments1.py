import cv2
import numpy as np

def edge_detection():
    # Initialize the camera (change the index if it's not 0)
    camera = cv2.VideoCapture(1)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Set the resolution (optional)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to capture image.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 100, 200)

        # Display the edges
        cv2.imshow("Edge Detection", edges)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()


def detect_objects():
    # Initialize the camera (change the index if necessary)
    camera = cv2.VideoCapture(1)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Set the resolution (optional)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to capture image.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_frame, 100, 200)

        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask to draw on
        output_frame = frame.copy()

        for contour in contours:
            # Filter small contours (noise)
            if cv2.contourArea(contour) < 300:
                continue

            # Draw the contours in red
            cv2.drawContours(output_frame, [contour], -1, (0, 0, 255), 2)

            # Calculate the center of the object
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Mark the center with a circle and display coordinates
                cv2.circle(output_frame, (cX, cY), 5, (255, 0, 0), -1)
                cv2.putText(output_frame, f"({cX}, {cY})", (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                            2)

        # Show the result
        cv2.imshow("Object Detection", output_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

def detect_better():
    pass


if __name__ == "__main__":
    detect_objects()
