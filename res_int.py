"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline as us


lengths = np.array([0, 0, 1.04, 1.82, 3.4, 6.32, 8.9, 11.71, 14.44, 15.00, 15.75])[::-1]
# lengths = np.linspace(1, 0, 20)
diameters = np.array([0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])[::-1]
# diameters = 1 / np.linspace(0.00000001, 1, 20)
inverse_diameters = 1 / diameters

reference_y_coord = 0.2


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


def bisectObjFunc(ycoord, args):
    """
    Finds a line which bisects the integral area.
    """

    area_curve = args[0]
    d_inverse_diameters = args[1]
    d_lengths = args[2]

    # get equation of line.
    m, c = get_line_eqn([0,0], [reference_y_coord, ycoord])

    # get straight line vals for all 1/D vals
    d_linear_lengths = linear(d_inverse_diameters, m, c)
    # plt.plot(d_inverse_diameters, d_linear_lengths)

    # take only the section where the difference between the two lines is +ve
    difference = d_lengths - d_linear_lengths
    r_difference = np.where(difference > 0, difference, np.zeros_like(difference))
    r_inverse_diameters = np.where(difference > 0, d_inverse_diameters, np.zeros_like(difference))

    # find area between line and curve on both sides
    a_between = abs(np.trapz(r_difference, r_inverse_diameters))
    # this is the measure of how well the line bisects the resolution curve
    non_bisectionality = abs((2 * a_between) - area_curve)
    return non_bisectionality


def rectSides(bisecting_coords, curve_area):
    # find the area of the rectangle from the distance along
    # the bisecting line passed to this function.

    x_line = bisecting_coords[0][0] - bisecting_coords[1][0]
    y_line = bisecting_coords[0][1] - bisecting_coords[1][1]

    side_ratio = x_line / y_line

    x_side = np.sqrt(curve_area * side_ratio)
    y_side = np.sqrt(curve_area / side_ratio)

    return [x_side, y_side]

def main():

    # new grid
    d_inverse_diameters = np.linspace(inverse_diameters[0], inverse_diameters[-1], int(1E4))

    # interpolate a denser resolution integral 
    interp = us(inverse_diameters, lengths, s=0)
    d_lengths = interp(d_inverse_diameters)

    area = abs(np.trapz(lengths, inverse_diameters))
    # we have the area under the curve, now need to find the dimensions
    # of a rectangle which matches this and bisects the curve

    ycoord_range = np.linspace(0, 5, 50)

    n_bs = []
    for i in ycoord_range:
        n_bs.append(bisectObjFunc(i, [area, d_inverse_diameters, d_lengths]))

    # find the y coordinate which gives the smallest non_bisectionality
    y_min = ycoord_range[np.argmin(n_bs)]

    # res_bisect = minimize(bisectObjFunc, [2], args=[area, d_inverse_diameters, d_lengths])
    # print(res_bisect.x)

    bisecting_coords = [[0,0], [reference_y_coord, y_min]]

    sides = rectSides(bisecting_coords, area)
    print(sides)
    plt.scatter(sides[0], sides[1])
    # plt.plot(ycoord_range, n_bs)
    plt.plot(d_inverse_diameters, d_lengths)

    plt.show()


main()
