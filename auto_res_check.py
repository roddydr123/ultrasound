"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from slice_thickness import extract_Ls
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


reference_x_coord = 0.01


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


def bisectObjFunc(ycoord, area_curve, inverse_diameters, lengths):
    """
    Finds a line which bisects the integral area.
    """

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
    ax.plot([0, sides[0]], [0, sides[1]], "k--")

    ax.plot(inverse_diameters, lengths, "k-", linewidth=2)
    # ax.scatter(inverse_diameters[:-2], lengths[:-2], c="k", linewidth=0.5, marker="o")
    ax.set_xlabel("$\\alpha$ (mm$^{-1}$)")
    ax.set_ylabel("L (mm)")

    ax.vlines(sides[0], 0, sides[1], colors=["k"], linestyles=["dotted"])
    ax.hlines(sides[1], 0, sides[0], colors=["k"], linestyles=["dotted"])

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, 1.1 * inverse_diameters[np.argmin(lengths)])

    offset = matplotlib.transforms.ScaledTranslation(0.05, 0, fig.dpi_scale_trans)
    ax.xaxis.get_majorticklabels()[0].set_transform(ax.xaxis.get_majorticklabels()[0].get_transform() + offset)

    plt.tight_layout()
    plt.locator_params(nbins=4)

    plt.show()


def calc_R(lengths:list, inverse_diameters:list, show=True):
    """Find the resolution integral, Dr, and Lr given visualisation lengths
    and inverse diameters.

    Args:
        lengths (list): List of distances over which each pipe could be visualised.
        inverse_diameters (list): Inverses of the pipe diameters.
        show (bool, optional): Whether to print info and show the L-alpha plot. Defaults to True.

    Returns:
        tuple: (Dr, Lr, R) The calculated results.
    """

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

    # new grid
    d_inverse_diameters = np.linspace(
        inverse_diameters[0], inverse_diameters[-1], int(1e4)
    )

    # interpolate a denser resolution integral
    interp = interp1d(inverse_diameters, lengths)
    d_lengths = np.array(interp(d_inverse_diameters))

    # find the area under the curve (the resolution integral)
    area = abs(np.trapz(lengths, inverse_diameters))

    res = minimize_scalar(bisectObjFunc, args=(area,d_inverse_diameters,d_lengths), method='Bounded', bounds=(0.001, 1000))

    y_min = res.x

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
