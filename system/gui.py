from matplotlib import pyplot as plt

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
    def __init__(self):
        self.distances = [0]
        self.real_distances = [0]
        self.moving_average = [0]
        self.times = [0]
        self.FRAMESIZE = 300
        self.threshold = 0.2
        self.previous_distance = 0
        self.before_previous_distance = 0
        self.MOVING_AVERAGE_FRAMESIZE = 5

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

