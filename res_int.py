"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from scipy.optimize import minimize


lengths = np.array([0, 0, 1.04, 1.82, 3.4, 6.32, 8.9, 11.71, 14.44, 15.00, 15.75])
diameters = np.array([0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])


def objectiveFunc(x):
    """
    (function) objectiveFunc: (x: tuple of 4 points describing rectangle) 

    Finds a line which bisects the integral area and a rectangle
    bisected by that line with the same area.
    """

    # get diagonal from rectangle, simply the 1st and 3rd coordinates.
    line_coords = [x[0], x[2]]

    # how does line split integral area?



def main():
    x = 1 / diameters

    area = abs(simpson(lengths, x, even="avg"))
    # we have the area under the curve, now need to find the dimensions
    # of a rectangle which matches this and bisects the curve
    print(area)

    plt.plot(x, lengths)
    plt.show()


main()
