from videos import Video
import numpy as np
from scipy.interpolate import UnivariateSpline as us
import matplotlib.pyplot as plt
from itertools import pairwise
from scipy.ndimage import gaussian_filter1d as gf1d


PATH = "/home/david/Documents/uni/year-5/ultrasound/"


def get_slice_thickness():
    viddata = {
               "filepath": f"{PATH}videos/", "filenumber": "89"}
    vid = Video(viddata)

    vid.save_slice_thickness_data(5, f"{PATH}scripts/analysed/vid{viddata['filenumber']}.txt")


def solve_for_x(x_points, y_points, y):

    xs = []

    # Find the interval containing the solution
    for i in range(len(x_points) - 1):
        if y_points[i] <= y <= y_points[i + 1] or y_points[i] >= y >= y_points[i + 1]:
            xs.append((x_points[i] + x_points[i + 1]) / 2)
    
    return xs


def extract_Ls():
    """Takes in videos for a probe and finds the depth range for which a series of
    slice thicknesses are larger.
    """

    required_videos = [45]
    # range of slice thicknesses to find L for.
    pipe_diameters = [4] # np.linspace(2, 8, 1)

    vid_arrays = []
    # load in slice thickness plot data
    for vid_number in required_videos:
        vid_path = f"{PATH}scripts/analysed/vid{vid_number}.txt"
        dataset = np.genfromtxt(vid_path, dtype=float, delimiter=",").T

        # sort and convert to mm
        ind = np.argsort(dataset[0])
        depths = np.array(dataset[0])[ind] * 10
        slice_thicknesses = np.array(dataset[1])[ind] * 10

        # average slice thicknesses recorded for the same depth.
        reduced_st = []
        reduced_depths, indices = np.unique(depths, return_index=True)
        pairs = list(pairwise(indices))
        for pair in pairs:
            reduced_st.append(np.average(slice_thicknesses[pair[0]:pair[1]]))
        # pairwise misses out the last section so add it back in.
        reduced_st.append(np.average(slice_thicknesses[list(pairs)[-1][1]:]))

        # smooth
        reduced_st = gf1d(reduced_st, 4)

        vid_arrays.append([reduced_depths, reduced_st])

        # plt.plot(reduced_depths, reduced_st)
        # plt.scatter(depths, slice_thicknesses)
        # plt.show()

    # y = 4
    # x = solve_for_x(reduced_depths, reduced_st, y)
    # print("The value of x for which y =", y, "is", x)

    # dense_xs = np.linsapce
    

    for diameter in pipe_diameters:
        # find the depth from each video
        for vid_data in vid_arrays:
            depth_vals = solve_for_x(vid_data[0], vid_data[1], diameter)
            print(depth_vals)
    


# get_slice_thickness()
extract_Ls()