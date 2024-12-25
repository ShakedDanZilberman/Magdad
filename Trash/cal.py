class LaserPointer:
    def __init__(self):
        self.point = (0, 0)
        self.board = Arduino("COM6")

        # Define the pins for the servos and the laser
        servoV_pin = 5
        servoH_pin = 3
        laser_pin = 8
        self.board.digital[laser_pin].write(1)
        # Attach the servo to the board
        self.servoV = self.board.get_pin(f"d:{servoV_pin}:s")  # 's' means it's a servo
        self.servoH = self.board.get_pin(f"d:{servoH_pin}:s")

        # Start an iterator thread to read analog inputs
        it = util.Iterator(self.board)
        it.start()
        # Coefficients for angleX polynomial
        self.AX = 113.9773
        self.BX = -0.0588
        self.CX = 0.0001
        self.DX = 0
        self.EX = -0.1736
        self.FX = 0.0001
        self.GX = 0.0000
        self.HX = 0.0001
        self.IX = -0.0000
        self.JX = 0

        # Coefficients for angleY polynomial
        self.AY = 69.3912
        self.BY = -0.1502
        self.CY = 0.0001
        self.DY = 0
        self.EY = 0.0144
        self.FY = 0.0
        self.GY = 0.0000
        self.HY = 0.0000
        self.IY = 0.0000
        self.JY = 0

        self.STARTX = 60
        self.STARTY = 40
        self.deltaX = 30
        self.deltaY = 20
        self.NUMITER = 10

    def calibration():
        mx, my = 0, 0

        def click_event(event, x, y, flags, param):
            global mx, my
            if event == cv2.EVENT_LBUTTONDOWN:
                # print(f"Clicked coordinates: {relative_x}, {relative_y}")
                mx, my = x, y

        def find_red_point(frame):
            """
            Finds the (x, y) coordinates of the single red point in the image.

            Args:
                frame: The input image (BGR format).

            Returns:
                A tuple (x, y) representing the center of the red point if found, or None if no red point is detected.
            """
            # Convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for red in HSV
            lower_red1 = np.array([0, 120, 70])  # Lower range for red
            upper_red1 = np.array([10, 255, 255])  # Upper range for red
            lower_red2 = np.array([170, 120, 70])  # Lower range for red (wrapping around 180 degrees)
            upper_red2 = np.array([180, 255, 255])  # Upper range for red

            # Create masks for red
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)  # Combine masks

            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return (0, 0)  # No red point found

            # Assume the largest contour is the red point (adjust as needed)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the center of the red point
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return (0, 0)  # Avoid division by zero

            cX = int(M["m10"] / M["m00"])  # x-coordinate
            cY = int(M["m01"] / M["m00"])  # y-coordinate

            return cX, cY

    def angle_calc(self, coordinates):
        X = coordinates[0]  # rx
        Y = coordinates[1]  # ry
        # Calculate angleX using the full polynomial expression
        angleX = (
                self.AX
                + self.BX * Y
                + self.CX * Y ** 2
                + self.DX * Y ** 3
                + self.EX * X
                + self.FX * X * Y
                + self.GX * X * Y ** 2
                + self.HX * X ** 2
                + self.IX * X ** 2 * Y
                + self.JX * X ** 3
        )
        # Calculate angleY using the full polynomial expression
        angleY = (
                self.AY
                + self.BY * Y
                + self.CY * Y ** 2
                + self.DY * Y ** 3
                + self.EY * X
                + self.FY * X * Y
                + self.GY * X * Y ** 2
                + self.HY * X ** 2
                + self.IY * X ** 2 * Y
                + self.JY * X ** 3
        )
        return angleX, angleY

    def move(self, point):
        self.point = point
        angleX, angleY = self.angle_calc(point)
        print(angleX, angleY)
        self.servoH.write(angleX)
        self.servoV.write(angleY)


class DecisionMaker:
    @staticmethod
    def avg_heat_maps(changes_map, contours_map):
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
            if np.mean(changes_map) < 20:
                return contours_map
            if np.mean(contours_map) < 20:
                return changes_map
            result = changes_map + contours_map
            result = np.clip(result, 0, 255)
            return result
        if isinstance(changes_map, np.ndarray) and changes_map.size > 1:
            return changes_map
        if isinstance(contours_map, np.ndarray) and contours_map.size > 1:
            return contours_map
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)


def show_targets(average):
    average = cv2.cvtColor(average, cv2.COLOR_GRAY2BGR)
    circles_high, circles_low, centers = ImageParse.generate_targets(average)
    for circle in circles_low:
        cv2.circle(
            average,
            (int(circle[0][0]), int(circle[0][1])),
            int(circle[1]),
            (0, 255, 0),
            1,
        )
    for circle in circles_high:
        cv2.circle(
            average,
            (int(circle[0][0]), int(circle[0][1])),
            int(circle[1]),
            (0, 0, 255),
            1,
        )
    for center in centers:
        cv2.circle(average, center, radius=1, color=(255, 0, 0), thickness=-1)
    cv2.imshow("average", average)
    return circles_high, circles_low, centers


def laser_thread():
    global centers
    laser_pointer = LaserPointer()
    while True:
        for center in centers:
            laser_pointer.move(center)


def main():
    global CAMERA_INDEX, timestep
    CameraIO.detectCameras()
    cam = cv2.VideoCapture(CAMERA_INDEX)
    rawHandler = RawHandler()
    newPixelsHandler = NewPixelsHandler()
    differenceHandler = DifferenceHandler()
    contoursHandler = ContoursHandler()
    accumulator_contours = Accumulator()

    number_of_frames = 0
    while True:
        number_of_frames += 1
        ret_val, img = cam.read()
        img = ImageParse.toGrayscale(img)

        if not ret_val:
            print("Camera @ index 1 not connected")
            CAMERA_INDEX = int(
                input("Enter the index of the camera you want to connect to: ")
            )
            cam = cv2.VideoCapture(CAMERA_INDEX)
            ret_val, img = cam.read()
            if not ret_val:
                print("Failed to connect to camera")
                break

        # Downsize and grayscale the image to better simulate IR camera - Remove this when IR camera is connected
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = ImageParse.toGrayscale(img)

        for handler in [
            rawHandler,
            newPixelsHandler,
            differenceHandler,
            contoursHandler,
        ]:
            handler.add(img)
            handler.display(img)

        # accumulator_changes.add(newPixelsHandler.get(), 0.9, 0.1)
        # changes_heat_map = newPixelsHandler.get()
        accumulator_contours.add(contoursHandler.get(), 0.99, 0.01)
        # show accumulated heatmaps
        # cv2.imshow("acc changes", accumulator_changes.get())
        cv2.imshow("acc contours", accumulator_contours.get())

        average = DecisionMaker.avg_heat_maps(newPixelsHandler.get(img), contoursHandler.get())
        cv2.imshow("newPixelsHandler.get()", newPixelsHandler.get(img))
        cv2.imshow("average before", average)
        circles_high, circles_low, centers = show_targets(average=average)

        if cv2.waitKey(1) == 32:  # Whitespace
            newPixelsHandler.clear()

        # Press Escape to exit
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
