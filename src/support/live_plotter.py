import matplotlib.pyplot as plt
import time
import random

class LivePlotter:
    def __init__(self, title="Live Plot", xlabel="X", ylabel="Y"):
        self.x_data = []
        self.y_data = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def update(self, y, x=None):
        """
        Add new data point(s) and update plot.
        - y must be provided
        - if x is None, auto-generate as index counter
        """
        if x is None:
            # Single value case
            if isinstance(y, (int, float)):
                self.y_data.append(y)
                self.x_data.append(len(self.y_data) - 1)
            else:
                # Iterable case
                self.y_data.extend(y)
                self.x_data.extend(range(len(self.x_data), len(self.x_data) + len(y)))
        else:
            # Both x and y provided
            if isinstance(y, (int, float)):
                self.x_data.append(x)
                self.y_data.append(y)
            else:
                self.x_data.extend(x)
                self.y_data.extend(y)

        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

import csv
class LivePlotter4:
    def __init__(self, titles=None, xlabels=None, ylabels=None):
        # Defaults
        if titles is None:
            titles = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"]
        if xlabels is None:
            xlabels = ["X1", "X2", "X3", "X4"]
        if ylabels is None:
            ylabels = ["Y1", "Y2", "Y3", "Y4"]

        # Data buffers for 4 subplots
        self.x_data = [[] for _ in range(4)]
        self.y_data = [[] for _ in range(4)]

        # Figure + axes
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.axs = self.axs.flatten()

        # Create line objects
        self.lines = []
        for i, ax in enumerate(self.axs):
            line, = ax.plot([], [], '-')
            ax.set_title(titles[i])
            ax.set_xlabel(xlabels[i])
            ax.set_ylabel(ylabels[i])
            self.lines.append(line)

        plt.tight_layout()

    def _append_data(self, idx, y, x):
        """Internal helper for adding data to subplot idx."""
        if x is None:
            # Auto-generate x
            if isinstance(y, (int, float)):
                self.y_data[idx].append(y)
                self.x_data[idx].append(len(self.y_data[idx]) - 1)
            else:
                n = len(y)
                start = len(self.x_data[idx])
                self.y_data[idx].extend(y)
                self.x_data[idx].extend(range(start, start + n))
        else:
            # x provided
            if isinstance(y, (int, float)):
                self.x_data[idx].append(x)
                self.y_data[idx].append(y)
            else:
                self.x_data[idx].extend(x)
                self.y_data[idx].extend(y)

    def update(self, y_list, x_list=None):
        """
        Update all 4 subplots.
        - y_list: list/tuple of 4 values or iterables
        - x_list: list/tuple of 4 values or iterables, or None
        """
        if x_list is None:
            x_list = [None] * 4

        for i in range(4):
            self._append_data(i, y_list[i], x_list[i])

            # Update line
            self.lines[i].set_xdata(self.x_data[i])
            self.lines[i].set_ydata(self.y_data[i])

            # Rescale
            ax = self.axs[i]
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    def save_csv(self, prefix="plot"):
        """
        Save each subplot's data to separate CSV files:
        prefix_0.csv, prefix_1.csv, prefix_2.csv, prefix_3.csv
        """
        for i in range(4):
            filename = f"{prefix}_{i}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for x, y in zip(self.x_data[i], self.y_data[i]):
                    writer.writerow([x, y])
            print(f"Saved {filename}")


import numpy as np
import time

# Example usage
if __name__ == "__main__":
    # plotter = LivePlotter(title="Y-driven Plot")
    #
    # # Case 1: provide only y, x auto-generated
    # for i in range(10):
    #     plotter.update(random.random())
    #     time.sleep(0.05)
    #
    # # Case 2: provide both x and y
    # for i in range(10, 20):
    #     plotter.update(random.random(), x=i)
    #     time.sleep(0.05)
    #
    # plt.show()

    plotter = LivePlotter4()

    for t in range(200):
        y1 = np.sin(t * 0.1)
        y2 = np.cos(t * 0.1)
        y3 = np.tan(t * 0.05)
        y4 = np.sin(t * 0.1) * np.cos(t * 0.1)

        plotter.update([y1, y2, y3, y4])
        # time.sleep(0.01)
