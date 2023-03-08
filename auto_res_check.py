"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
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


def calc_R(lengths, inverse_diameters, show=True):

    # extrapolate to x axis and close integral

    # find smallest detectable pipe index
    s_index = np.argmin(lengths) - 1


    extrap_inverse_diam = inverse_diameters[s_index] - (
        lengths[s_index]
        * (inverse_diameters[s_index - 1] - inverse_diameters[s_index])
        / (lengths[s_index - 1] - lengths[s_index] + 0.00001)
        + 0.00001
    )

    # set the biggest invisible pipe to whichever's smaller out of the extrapolated inverse diameter
    # or its usual inverse diameter.
    if inverse_diameters[s_index + 1] > extrap_inverse_diam:
        inverse_diameters[s_index + 1] = extrap_inverse_diam

    lengths[s_index + 1 :] = 0

    # # new grid
    # inverse_diameters = np.linspace(
    #     inverse_diameters[0], inverse_diameters[-1], int(1e4)
    # )

    # # interpolate a denser resolution integral
    # interp = us(inverse_diameters, lengths, s=0, k=1)
    # lengths = np.array(interp(inverse_diameters))

    # make sure the splines arent bouncing around in the negatives or coming back from 0.
    negative = np.where(lengths >= 0, lengths, np.zeros_like(lengths))
    first_zero = np.argmin(negative)
    negative[first_zero:] = 0
    lengths = negative

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

    char_res = np.round(1/sides[0], 3)
    depth_field = np.round(sides[1], 2)
    R = np.round(area, 2)

    if show is True:
        print(f"characteristic resolution: {char_res} mm")
        print(f"depth of field: {depth_field} mm")
        print(f"Resolution integral: {R}")
        plotter(sides, inverse_diameters, lengths)
    return char_res, depth_field, R


def setup():
    videos = ["58", "59", "62", "63"]
    # videos = ["27", "28", "29", "30"]
    # videos = ["01", "04"]
    # videos = [25,26,31,32]
    # videos = [35,36,37,38] # 18L6 first go
    # videos = [68,69,70,71] # 18L6 diff probe
    # videos = [64,65,66,67]
    # videos = [78, 79, 80, 81]
    videos = [50, 51, 56, 57]

    # choose inverse diameter range
    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters[::-1]

    L_dict, lengths, diameters = extract_Ls(videos, pipe_diameters, 20, 3)

    diameters = np.array(diameters)[::-1]
    inverse_diameters = 1 / diameters

    lengths = np.array(lengths)[::-1]

    calc_R(lengths, inverse_diameters)


def all_vids():
    videos = [
    ["58", "59", "62", "63"], # 14L-5
    [60, 61, 62, 63],   # 14L-5
    ["27", "28", "29", "30"], # 9L
    ["01", "04"], # 9L
    [25,26,31,32],  # C15
    [35,36,37,38],  # 18L6
    [68,69,70,71],  # 18L6
    [64,65,66,67],  # 4C1
    [78, 79, 80, 81],   # 6C1
    [21,22,23,24],  # 9L4
    [72,73,74,75],  #9L4
    [50, 51, 56, 57] # ML6-15
    ]

    names = ["14L-5", "14L-5", "9L", "9L", "C1-5", "18L-6", "18L-6", "4C1", "6C1", "9L4", "9L4", "ML6-15"]

    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters[::-1]

    vals = []
    # to_be_averaged = [1, 3]

    for i, video in enumerate(videos):
        print(names[i])

        if i == 1 or i == 3:
            continue

        L_dict, lengths, diameters = extract_Ls(video, pipe_diameters, 20, 3)

        diameters = np.array(diameters)[::-1]
        inverse_diameters = 1 / diameters

        lengths = np.array(lengths)[::-1]

        data = calc_R(lengths, inverse_diameters, show=False)

        # specially for averaging 9L and 14L5
        if i == 0 or i == 2:
            L_dict, lengths, diameters = extract_Ls(videos[i+1], pipe_diameters, 20, 3)

            diameters = np.array(diameters)[::-1]
            inverse_diameters = 1 / diameters

            lengths = np.array(lengths)[::-1]

            data2 = calc_R(lengths, inverse_diameters, show=False)

            data = np.average((data, data2), axis=0)

        vals.append([names[i], "ST", data[2], data[1], data[0]])

    with open("analysed/gen3/auto_res_check.txt", "w") as f:
        for line in vals:
            f.write(str(line)[1:-1] + "\n")


if __name__=="__main__":
    all_vids()
    # setup()
