from matplotlib import pyplot as plt

class LIDARDistancesGraph:
    def __init__(self):
        self.distances = [0]
        self.real_distances = [0]
        self.moving_average = [0]
        self.times = [0]
        self.FRAMESIZE = 300
        self.threshold = 0.2
        self.previous_distance = 0
        self.before_previous_distance = 0

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
        self.moving_average.append(sum(self.distances[-5:]) / 5)

    def get_distances(self):
        return self.times, self.distances, self.real_distances
    
    def distance(self):
        return self.distances[-1]
    
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

