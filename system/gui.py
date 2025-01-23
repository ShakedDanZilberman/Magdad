from matplotlib import pyplot as plt
from constants import IMG_WIDTH, IMG_HEIGHT
import numpy as np
import threading
import cv2
import time

class LIDARDistancesGraph:
    """
    Class to plot the LIDAR distances in real-time.

    Attributes:
        distances (list[float]): List of filtered distances.
        real_distances (list[float]): List of raw distances.
        moving_average (list[float]): List of moving averages of the distances.
        times (list[float]): List of times at which the distances were measured.
        FRAMESIZE (int): Number of frames to display.
        threshold (float): Threshold for filtering out incorrect readings.
        previous_distance (float): The previous distance measured.
        before_previous_distance (float): The distance before the previous distance.
        MOVING_AVERAGE_FRAMESIZE (int): Number of frames to consider for the moving average.
        fig (plt.Figure): The figure to plot on.
        ax (plt.Axes): The axes to plot on.
        line (plt.Line2D): The line for the filtered distances.
        realline (plt.Line2D): The line for the raw distances.
        moving_average_line (plt.Line2D): The line for the moving averages.

    Methods:
        add_distance: Add a distance to the graph.
        get_distances: Get the distances from the graph.
        distance: Get the last distance.
        plot: Plot the graph.

    Logic:
        - The class uses a moving average to smooth out the distances.
        - The class filters out incorrect readings by comparing the current distance to the previous two distances.
    """
    def __init__(self, show=True):
        self.distances = [0]
        self.real_distances = [0]
        self.moving_average = [0]
        self.times = [0]
        self.FRAMESIZE = 300
        self.threshold = 0.2
        self.previous_distance = 0
        self.before_previous_distance = 0
        self.MOVING_AVERAGE_FRAMESIZE = 5

        self.show = show
        if show:
            # set the plt size to 1/2 of the default
            plt.rcParams['figure.figsize'] = [6.4, 4.8]
            plt.ion()
            self.fig, self.ax = plt.subplots()
            ax = self.ax
            self.line, = ax.plot([], [], 'b-')
            self.realline, = ax.plot([], [], 'r--')
            self.moving_average_line, = ax.plot([], [], 'g-')
            ax.set_xlabel("Time")
            ax.set_ylabel("LIDAR Distance")
            ax.set_title("LIDAR Distance vs Time")
            plt.legend(["Filtered", "Raw", "Moving Average"])
            plt.show()

    def add_distance(self, distance):
        self.real_distances.append(distance)
        self.times.append(self.times[-1] + 1)

        if len(self.real_distances) > self.FRAMESIZE:
            self.real_distances.pop(0)
            self.times.pop(0)
            self.distances.pop(0)
            self.moving_average.pop(0)

        correctReading = False
        if abs(self.previous_distance - distance) < self.threshold * distance:
            correctReading = True
        if abs(self.before_previous_distance - distance) < self.threshold * distance:
            correctReading = True
        self.before_previous_distance = self.previous_distance
        self.previous_distance = distance
        self.distances.append(self.distances[-1] if not correctReading else self.real_distances[-1])

        # add the average of the last 5 readings to the moving average list
        self.moving_average.append(sum(self.distances[-self.MOVING_AVERAGE_FRAMESIZE:]) / self.MOVING_AVERAGE_FRAMESIZE)

    def get_distances(self):
        return self.times, self.distances, self.real_distances, self.moving_average
    
    def distance(self):
        return self.moving_average[-1]
    
    def plot(self):
        if not self.show:
            return
        self.line.set_xdata(self.times)
        self.line.set_ydata(self.distances)
        self.realline.set_xdata(self.times)
        self.realline.set_ydata(self.real_distances)
        self.moving_average_line.set_xdata(self.times)
        self.moving_average_line.set_ydata(self.moving_average)
        try:
            self.ax.relim()
        except ValueError:
            pass
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

class GUI:
    def __init__(self):
        self.img = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
        self.targets = []
        self.changes = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
        self.contours = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)

    def add(self, img, targets, changes, contours, circles_low, circles_high, yolo, yolo_centers):
        self.img = img
        self.targets = targets
        self.changes = changes
        self.contours = contours
        self.circles_low = circles_low
        self.circles_high = circles_high
        self.yolo = yolo
        self.yolo_centers = yolo_centers

    def display(self):
        # merge_image should have four quadrents, each one the size of self.img
        merged_image = np.zeros((2*IMG_HEIGHT, 2*IMG_WIDTH), np.uint8)
        zeros_frame = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
        # convert to color image
        merged_image = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2BGR)
        # top left
        if self.img is not None:
            frame = self.img.copy()
        else:
            self.img = zeros_frame.copy()
        if self.img is not None:
            # (as grayscale)
            merged_image[:IMG_HEIGHT, :IMG_WIDTH, 0] = frame
            merged_image[:IMG_HEIGHT, :IMG_WIDTH, 1] = frame
            merged_image[:IMG_HEIGHT, :IMG_WIDTH, 2] = frame
        # add the targets as red circles
        for target in self.targets:
            red = (0, 0, 255)
            target_size = 2
            cv2.circle(merged_image, (int(target[0]), int(target[1])), target_size, red, -1)
            LOW_COLOR = (0, 255, 0)
            HIGH_COLOR = (0, 0, 255)
            for circle in self.circles_low:
                cv2.circle( 
                    merged_image,
                    (int(circle[0][0]), int(circle[0][1])),
                    int(circle[1]),
                    LOW_COLOR,
                    1,
                )
            for circle in self.circles_high:
                cv2.circle(
                    merged_image,
                    (int(circle[0][0]), int(circle[0][1])),
                    int(circle[1]),
                    HIGH_COLOR,
                    1,
                )
        print("Number of targets:", len(self.targets))
        # top right
        if self.changes is not None:
            # (in green)
            merged_image[:IMG_HEIGHT, IMG_WIDTH:2*IMG_WIDTH, 1] = self.changes
            merged_image[:IMG_HEIGHT, IMG_WIDTH:2*IMG_WIDTH, 0] = zeros_frame
            merged_image[:IMG_HEIGHT, IMG_WIDTH:2*IMG_WIDTH, 2] = zeros_frame
            cv2.putText(merged_image, " Changes", (IMG_WIDTH, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # bottom left
        if self.contours is not None:
            # (in blue)
            merged_image[IMG_HEIGHT:2*IMG_HEIGHT, :IMG_WIDTH, 0] = self.contours
            merged_image[IMG_HEIGHT:2*IMG_HEIGHT, :IMG_WIDTH, 1] = zeros_frame
            merged_image[IMG_HEIGHT:2*IMG_HEIGHT, :IMG_WIDTH, 2] = zeros_frame
            cv2.putText(merged_image, " Contours", (0, IMG_HEIGHT + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # bottom right
        if self.yolo is not None:
            # (in red)
            merged_image[IMG_HEIGHT:2*IMG_HEIGHT, IMG_WIDTH:2*IMG_WIDTH, 2] = self.yolo
            merged_image[IMG_HEIGHT:2*IMG_HEIGHT, IMG_WIDTH:2*IMG_WIDTH, 0] = zeros_frame
            merged_image[IMG_HEIGHT:2*IMG_HEIGHT, IMG_WIDTH:2*IMG_WIDTH, 1] = zeros_frame
            cv2.putText(merged_image, " YOLO", (IMG_WIDTH, IMG_HEIGHT + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for center in self.yolo_centers:
                red = (100, 0, 255)
                target_size = 2
                cv2.circle(merged_image, (int(center[0]), int(center[1])), target_size, red, -1)


        cv2.imshow("CounterStrike Magdad", merged_image)
        time.sleep(0.1)
