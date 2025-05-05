import cv2
import numpy as np
from math import atan2, degrees


def coordinate_to_angle(x, y):
    """
    Convert coordinates to angle relative to the constant coordinates GUN in degrees.
    """
    # Calculate the angle in radians
    angle_rad = atan2(y - GUN[1], x - GUN[0])

    # Convert to degrees
    angle_deg = degrees(angle_rad)

    # Normalize the angle to be between 0 and 360 degrees
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def show_camera_feed_from_2_cameras(CAMERA_INDEX_0_1, CAMERA_INDEX_1):
    """
    Show camera feed from two cameras side by side.
    """
    cam1 = cv2.VideoCapture(CAMERA_INDEX_0_1)
    cam2 = cv2.VideoCapture(CAMERA_INDEX_1)

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            print("Error: Unable to read from one of the cameras.")
            break

        # Resize frames to the same height
        height = min(frame1.shape[0], frame2.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * (height / frame1.shape[0])), height))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * (height / frame2.shape[0])), height))

        # Combine frames horizontally
        combined_frame = np.hstack((frame1, frame2))

        cv2.imshow("Camera Feed", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

def show_camera_feed_from_3_cameras(CAMERA_INDEX_0_1, CAMERA_INDEX_1, CAMERA_INDEX_0_3):
    """
    Show camera feed from three cameras side by side.
    """
    cam1 = cv2.VideoCapture(CAMERA_INDEX_0_1)
    cam2 = cv2.VideoCapture(CAMERA_INDEX_1)
    cam3 = cv2.VideoCapture(CAMERA_INDEX_0_3)

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        ret3, frame3 = cam3.read()

        if not ret1 or not ret2 or not ret3:
            print("Error: Unable to read from one of the cameras.")
            break

        # Resize frames to the same height
        height = min(frame1.shape[0], frame2.shape[0], frame3.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * (height / frame1.shape[0])), height))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * (height / frame2.shape[0])), height))
        frame3 = cv2.resize(frame3, (int(frame3.shape[1] * (height / frame3.shape[0])), height))

        # Combine frames horizontally
        combined_frame = np.hstack((frame1, frame2, frame3))

        cv2.imshow("Camera Feed", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam1.release()
    cam2.release()
    cam3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera_feed_from_3_cameras(2, 1, 3)  # Replace with your camera indices

