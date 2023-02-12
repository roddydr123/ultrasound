"""
This is a script for calculating the resolution integral from measurements
made with an ultrasound machine and Edinburgh Pipe Phantom (EPP).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as us


"""Visualisation lengths from my EPP measurements in cm"""
# lengths = np.array([0, 0, 1.04, 1.82, 3.4, 6.32, 8.9, 11.71, 14.44, 15.00, 15.75])    # 9L first measurement 18/10/22
# lengths = np.array([0, 0, 1.01, 3.34, 5.03, 6.21, 9.37, 12.17, 15.26, 15.73, 16.41])  # 9L second, 25/10/22
# lengths = np.array([2.83, 3.24, 4.41, 5.05, 7.44, 8.12, 8.62, 9.47, 9.59, 9.93, 10.26])   # ML6-15 25/10/22

"""Visualisation lengths from slice thickness videos in cm"""
# lengths = np.array([0.0, 3.75, 6.3, 8.1, 8.1, 10, 13, 13, 13, 13, 13, 13, 13])     # 9L st videos in notes
# lengths = np.array([0.0, 3.75, 6.3, 8.1, 8.1, 10, 15.2, 18.5, 18.5, 18.5, 18.5, 18.5, 18.5])     # adjusted LCP 9L
# lengths = np.array([0.0, 4, 5.3, 7.8, 7.9, 8, 8, 8, 8, 8, 8, 8, 8]) # ML615 st videos in notes
# lengths = np.array([0.0, 4, 5.3, 9, 9.4, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2, 10.2]) # adjusted ML615
# lengths = np.array([0.0, 3.6, 3.9, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]) # 14L5 st videos in notes
# lengths = np.array([0.0, 3.6, 3.9, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5]) # adjusted 14L5
# lengths = np.array([0.0, 4, 6, 8.7, 9.8, 11, 11, 11, 11, 11, 11, 11, 11]) # 9L4 st videos in notes
# lengths = np.array([0.0, 4, 6, 8.7, 9.8, 13.8, 13.8, 13.8, 13.8, 13.8, 13.8, 13.8, 13.8])   # adjusted 9L4
# lengths = np.array([0.0, 0.0, 3.75, 7.6, 7.75, 10.1, 15.05, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5])    # C15 st videos in notes
# lengths = np.array([0.0, 0.0, 3.75, 7.6, 7.75, 10.1, 15.05, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8])    # adjusted C15
# lengths = np.array([0.0, 3.1, 5.8, 6.9, 7.7, 10, 12.7, 13, 13, 13, 13, 13, 13]) # 9L repeat st videos in notes
# lengths = np.array([0.0, 3.1, 5.8, 6.9, 7.7, 10, 12.7, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8]) # adjusted 9L repeat

"""Visualisation lengths from slice thickness videos in mm"""
# lengths = [0,0,0,0,0,0,0,10,38,64,96,96,185]    # 9L-D
# lengths = [0,0,0,0,0,0,18,22,42,58,94,95,102]   # ML6-15
# lengths = [0,0,0,0,0,18,21,32,39,61,65,65,70]   # 14L5
# lengths = [0,0,0,0,0,0,6.5,16,36,54,78,108,138] # 9L4
# lengths = [0,0,0,0,0,0,0,0,11,40,111,114,198]   # C1-5
# lengths = [0,0,0,0,0,0,0,16,36,67,84,84,180]    # 9L-D repeat
# lengths = [0,0,0,0,0,0,24,25,56,65,65,65,72]    # 18L6
# lengths = [0,0,0,0,0,0,0,20,31,52,88,88,95]     # ML6-15 repeat
# lengths = [0,0,0,0,0,14,18,33,36,63,63,63,68]   # 14L5 repeat
# lengths = [0,0,0,0,0,0,0,0,0,12,58,127,160]     # 4C1
# lengths = [0,0,0,0,0,11,26,29,57,57,57,57,62]   # 18L6 different probe
# lengths = [0,0,0,0,0,0,7,43,46,61,69,125,134]   # 9L4 different probe
# lengths = [0,0,0,0,0,0,0,0,1,82,152,152,168]      # 6C1

"""Visualisation lengths from NHS data spreadsheet or lab folder in mm"""
# lengths = np.array([0.0, 2.9, 26.0, 40.3, 55.0, 71.0, 81.3, 89.7, 93.4, 94.6, 97.2, 98.6]) / 10   # NHS data for ML6-15 31/03/15
# lengths = np.array([0.0,0.0,3.8,18.1,23.9,70.1,85.1,128.6,147.0,165.6,165.1, 192.0]) / 10 # NHS data for 9LD 01/04/15
# lengths = np.array([0.0,0.0,28.2,44.0,53.2,73.0,85.2,91.9,95.5,96.2,96.2,96.2]) / 10    # NHS data ML6-15 20/08/13
# lengths = np.array([0.0, 2.0, 10.8, 22.7, 39.6, 44.9, 57.0, 65.8, 69.3, 72.5, 75.8, 76.8]) / 10 # NHS data 14L5 21/07/10
# lengths = np.array([0.0, 0.0, 6.4, 24.4, 43.5, 58.5, 83.9, 97.7, 132.6, 137.9, 137.9, 137.9]) / 10  # NHS data 9L4 lab folder
# lengths = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 41.0, 73.6, 121.0, 169.5, 197.1, 203.1, 221.5]) / 10   # NHS data C1-5 lab folder
# lengths = [0,2.2,12.8,19.1,28.7,39.7,60.9,64.9,65.5,66.6,65.9,65.2] # NHS data in folder 18L6
# lengths = [0,0,0,0,0,4.6,47.1,117.2,175.6,195,200.5,215.8]  # NHS data in folder 6C1
lengths = [0,0,0,0,0,22.9,72.6,108,148,197,201,233] # 4C1 NHS folder


"""Pipe/slice thickness diameters in mm"""
# diameters = np.array([0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])
# diameters = [0.15, 0.3, 0.4, 0.6, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]    # my custom diameters for ST
# diameters = np.array([0.05, 0.3, 0.4, 0.6, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])    # my custom diameters for 14L5 ST
# diameters = np.array([0.35, 0.42, 0.56, 0.70, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.9, np.inf])   # NHS data teams spreadsheet
diameters = np.array([0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])   # NHS data lab folder

diameters = np.array(diameters)[::-1] / np.sqrt(np.cos(np.deg2rad(40))) # convert to effective diameter and reverse
inverse_diameters = 1 / diameters
lengths = np.array(lengths)[::-1]

reference_x_coord = 0.00002


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
    ax.set_xlabel("$\\alpha$ (1/mm)")
    ax.set_ylabel("Lr (mm)")

    ax.vlines(sides[0], 0, sides[1], colors=["k"])
    ax.hlines(sides[1], 0, sides[0], colors=["k"])

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])

    plt.show()


def main():

    # extrapolate to x axis and close integral

    # find smallest detectable pipe index
    s_index = np.argmin(lengths) - 1

    extrap_inverse_diam = inverse_diameters[s_index] - (lengths[s_index] * (inverse_diameters[s_index-1] \
                          - inverse_diameters[s_index])/(lengths[s_index-1] - lengths[s_index] + 0.00001)+0.00001)

    # set the biggest invisible pipe to whichever's smaller out of the extrapolated inverse diameter
    # or its usual inverse diameter.
    if inverse_diameters[s_index + 1] > extrap_inverse_diam:
        inverse_diameters[s_index + 1] = extrap_inverse_diam

    lengths[s_index + 1:] = 0

    # new grid
    d_inverse_diameters = np.linspace(inverse_diameters[0], inverse_diameters[-1], int(1E4))

    # interpolate a denser resolution integral 
    interp = us(inverse_diameters, lengths, s=0, k=1)
    d_lengths = np.array(interp(d_inverse_diameters))

    # make sure the splines arent bouncing around in the negatives or coming back from 0.
    negative = np.where(d_lengths >= 0, d_lengths, np.zeros_like(d_lengths))
    first_zero = np.argmin(negative)
    negative[first_zero:] = 0
    d_lengths = negative

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
        n_bs.append(bisectObjFunc(i, [area, d_inverse_diameters, d_lengths]))

    # find the y coordinate which gives the smallest non_bisectionality
    y_min = ycoord_range[np.argmin(n_bs)]

    # package up the best bisecting line coords and find the dimensions of
    # the rectangle which has the same integral and is also bisected by the line.
    # the sides are the characteristic resolution and the depth of field.
    bisecting_coords = [[0,0], [reference_x_coord, y_min]]
    sides = rectSides(bisecting_coords, area)

    print(f"characteristic resolution: {np.round(1/sides[0], 3)}mm")
    print(f"depth of field: {np.round(sides[1], 2)}mm")
    print(f"Resolution integral: {np.round(area, 2)}")

    plotter(sides, d_inverse_diameters, d_lengths)

main()