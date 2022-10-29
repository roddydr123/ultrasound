"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as us


# lengths = np.array([0, 0, 1.04, 1.82, 3.4, 6.32, 8.9, 11.71, 14.44, 15.00, 15.75])    # 9L first measurement 18/10/22
# lengths = np.array([0, 0, 1.01, 3.34, 5.03, 6.21, 9.37, 12.17, 15.26, 15.73, 16.41])  # 9L second, 25/10/22
lengths = np.array([2.83, 3.24, 4.41, 5.05, 7.44, 8.12, 8.62, 9.47, 9.59, 9.93, 10.26])   # ML6-15 25/10/22
# lengths = np.array([0.0, 2.9, 26.0, 40.3, 55.0, 71.0, 81.3, 89.7, 93.4, 94.6, 97.2, 98.6]) / 10   # NHS data for ML6-15 31/03/15
# lengths = np.array([0.0,0.0,3.8,18.1,23.9,70.1,85.1,128.6,147.0,165.6,165.1, 192.0]) / 10 # NHS data for 9LD 01/04/15
# lengths = np.array([0.0,0.0,28.2,44.0,53.2,73.0,85.2,91.9,95.5,96.2,96.2,96.2]) / 10    # NHS data ML6-15 20/08/13
diameters = np.array([0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])
# diameters = np.array([0.35, 0.42, 0.56, 0.70, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.9, np.inf])   # NHS data
diameters = diameters[::-1] / np.sqrt(np.cos(np.deg2rad(40))) # convert to effective diameter and reverse
inverse_diameters = 1 / diameters
lengths = lengths[::-1] * 10   # convert to mm

reference_x_coord = 0.02


class Line():
    def __init__(self, point1, point2, xvals) -> None:
        self.point1 = point1
        self.point2 = point2
        self.xvals = xvals

    def get_line_eqn(self):
        """
        Given two points, return the coefficients of the equation of the
        line between these points. ax + by = c
        """
        a = self.point2[1] - self.point1[1]
        b = self.point2[0] - self.point1[0]
        c = a * self.point1[0] + b * self.point2[0]

        m = a / b
        new_c = c / b
        return m, new_c

    def get_points_on_line(self):
        m, c = self.get_line_eqn()
        yvals = self.linear(self.xvals, m, c)
        return yvals

    def linear(self, x, m, c):
        return (m * x) + c


def bisectObjFunc(ycoord, args):
    """
    Finds a line which bisects the integral area.
    """

    area_curve = args[0]
    d_inverse_diameters = args[1]
    d_lengths = args[2]

    # get straight line vals for all 1/D vals
    line = Line([0,0], [reference_x_coord, ycoord], d_inverse_diameters)
    d_linear_lengths = line.get_points_on_line()
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


def plotter(sides, d_inverse_diameters, d_lengths):

    fig, ax = plt.subplots()
    ax.scatter(sides[0], sides[1], c="k", marker="x")

    ax.plot(d_inverse_diameters, d_lengths)
    ax.set_xlabel("Inverse diameter (1/mm)")
    ax.set_ylabel("Depth of field (mm)")

    ax.vlines(sides[0], 0, sides[1], colors=["k"])
    ax.hlines(sides[1], 0, sides[0], colors=["k"])

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])

    plt.show()


def main():

    # new grid
    d_inverse_diameters = np.linspace(inverse_diameters[0], inverse_diameters[-1], int(1E4))

    # interpolate a denser resolution integral 
    interp = us(inverse_diameters, lengths, s=0)
    d_lengths = interp(d_inverse_diameters)

    # find the area under the curve (the resolution integral)
    area = abs(np.trapz(lengths, inverse_diameters))

    # bisecting lines are parametrised by the y coordinate of a point at
    # reference_x_coord. This variable sets the range of y coords which
    # are explored.
    ycoord_range = np.linspace(0, 5, 50)

    # go through each bisecting line and find how well it bisects the
    # resolution integral.
    n_bs = []
    for i in ycoord_range:
        n_bs.append(bisectObjFunc(i, [area, d_inverse_diameters, d_lengths]))

    # find the y coordinate which gives the smallest non_bisectionality
    y_min = ycoord_range[np.argmin(n_bs)]

    # package up the best bisecting line coords and find the dimensions of
    # the rectangle which has the same integral and is also bisected by the line.
    # the sides are the characteristic resolution and the depth of field.
    bisecting_coords = [[0,0], [reference_x_coord, y_min]]
    sides = rectSides(bisecting_coords, area)

    print(f"characteristic resolution: {np.round(1/sides[0], 2)}mm")
    print(f"depth of field: {np.round(sides[1], 1)}mm")
    print(f"Resolution integral: {np.round(area, 1)}mm^2")

    plotter(sides, d_inverse_diameters, d_lengths)


main()
