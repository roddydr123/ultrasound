"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline as us


lengths = np.array([0, 0, 1.04, 1.82, 3.4, 6.32, 8.9, 11.71, 14.44, 15.00, 15.75])[::-1]
# lengths = np.array([1, 1, 0])#[::-1]
diameters = np.array([0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])[::-1]
# diameters = np.array([np.inf, 1, 0.99])#[::-1]
inverse_diameters = 1 / diameters


def get_line_eqn(p, q):
    """
    Given two points, return the coefficients of the equation of the
    line between these points. ax + by = c
    """
    a = q[1] - p[1]
    b = q[0] - p[0]
    c = a * p[0] + b * p[0]

    m = a / b
    new_c = c / b
    return m, new_c


def linear(x, m, c):
    return (m * x) + c


def objectiveFunc(rectangle_coords, args):
    """
    Finds a line which bisects the integral area and a rectangle
    bisected by that line with the same area.
    """

    area_curve = args[0]
    d_inverse_diameters = args[1]
    d_lengths = args[2]

    # get diagonal from rectangle, simply the 1st and 3rd coordinates.
    m, c = get_line_eqn(rectangle_coords[0], rectangle_coords[2])

    # get straight line vals for all 1/D vals
    d_linear_lengths = linear(d_inverse_diameters, m, c)
    plt.plot(d_inverse_diameters, d_linear_lengths)

    # find area between line and curve on both sides
    a_between = abs(np.trapz(d_lengths - d_linear_lengths, d_inverse_diameters))
    non_bisectionality = (2 * a_between) - area_curve
    print(a_between)
    print(non_bisectionality)


def main():

    # new grid
    d_inverse_diameters = np.linspace(inverse_diameters[0], inverse_diameters[-1], 40)

    # interpolate a denser resolution integral 
    interp = us(inverse_diameters, lengths, s=0)
    d_lengths = interp(d_inverse_diameters)

    area = abs(np.trapz(lengths, inverse_diameters))
    # we have the area under the curve, now need to find the dimensions
    # of a rectangle which matches this and bisects the curve
    print(area)

    objectiveFunc([(0,0),(1,0), (1,2), (0,1)], [area, d_inverse_diameters, d_lengths])
    # plt.plot(inverse_diameters, lengths)
    plt.plot(d_inverse_diameters, d_lengths)
    plt.show()


main()
