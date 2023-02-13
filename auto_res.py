import numpy as np
import matplotlib.pyplot as plt
from slice_thickness import extract_Ls


reference_x_coord = 0.00002


class Line:
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
    inverse_diameters = args[1]
    lengths = args[2]

    # get straight line vals for all 1/D vals
    line = Line([0, 0], [reference_x_coord, ycoord], inverse_diameters)
    d_linear_lengths = line.get_points_on_line()
    # plt.plot(inverse_diameters, d_linear_lengths)

    # take only the section where the difference between the two lines is +ve
    difference = lengths - d_linear_lengths
    r_difference = np.where(difference > 0, difference, np.zeros_like(difference))
    r_inverse_diameters = np.where(
        difference > 0, inverse_diameters, np.zeros_like(difference)
    )

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


def plotter(sides, inverse_diameters, lengths):

    # find where y goes to zero to rescale the plot
    ind = np.nonzero(lengths)
    inverse_diameters = inverse_diameters[ind]
    lengths = lengths[ind]

    fig, ax = plt.subplots()
    ax.scatter(sides[0], sides[1], c="k", marker="x")

    ax.plot(inverse_diameters, lengths)
    ax.set_xlabel("$\\alpha$ (1/mm)")
    ax.set_ylabel("Lr (mm)")

    ax.vlines(sides[0], 0, sides[1], colors=["k"])
    ax.hlines(sides[1], 0, sides[0], colors=["k"])

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])

    plt.show()


def calc_resolution_integral():
    # videos = ["58", "59", "62", "63"]
    # videos = ["27", "28", "29", "30"]
    # videos = ["01", "04"]
    # videos = [25,26,31,32]
    # videos = [35,36,37,38]
    # videos = [68,69,70,71]
    # videos = [64,65,66,67]
    videos = [78, 79, 80, 81]

    # choose inverse diameter range
    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters[::-1]

    L_dict, lengths, diameters = extract_Ls(videos, pipe_diameters, 20, 3)
    # print(L_dict)
    # plt.scatter(diameters, lengths)
    # plt.show()
    diameters = np.array(diameters)[::-1] / np.sqrt(
        np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse
    inverse_diameters = 1 / diameters

    lengths = np.array(lengths)[::-1]

    # find the area under the curve (the resolution integral)
    area = abs(np.trapz(lengths, inverse_diameters))

    # bisecting lines are parametrised by the y coordinate of a point at
    # reference_x_coord. This variable sets the range of y coords which
    # are explored.
    ycoord_range = np.linspace(0, 5, 12500)

    # go through each bisecting line and find how well it bisects the
    # resolution integral.
    n_bs = []
    for i in ycoord_range:
        n_bs.append(bisectObjFunc(i, [area, inverse_diameters, lengths]))

    # find the y coordinate which gives the smallest non_bisectionality
    y_min = ycoord_range[np.argmin(n_bs)]

    # package up the best bisecting line coords and find the dimensions of
    # the rectangle which has the same integral and is also bisected by the line.
    # the sides are the characteristic resolution and the depth of field.
    bisecting_coords = [[0, 0], [reference_x_coord, y_min]]
    sides = rectSides(bisecting_coords, area)

    print(f"characteristic resolution: {np.round(1/sides[0], 3)}mm")
    print(f"depth of field: {np.round(sides[1], 2)}mm")
    print(f"Resolution integral: {np.round(area, 2)}")

    # print(inverse_diameters, lengths)

    plotter(sides, inverse_diameters, lengths)


calc_resolution_integral()
