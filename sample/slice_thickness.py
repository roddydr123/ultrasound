from videos import Video
import numpy as np
import matplotlib.pyplot as plt
from itertools import pairwise
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.signal import find_peaks
import sys
import pathlib


p = pathlib.Path(__file__).parents
PATH = p[1]


def get_slice_thickness(number):
    viddata = {"filepath": f"{PATH}/videos/", "filenumber": f"{number}"}
    vid = Video(viddata)

    vid.save_slice_thickness_data(
        5, f"{PATH}/scripts/analysed/gen3/vid{viddata['filenumber']}.txt"
    )


def solve_for_x(x_points, y_points, y):

    xs = []

    # Find the interval containing the solution
    for i in range(len(x_points) - 1):
        if y_points[i] <= y <= y_points[i + 1] or y_points[i] >= y >= y_points[i + 1]:
            xs.append((x_points[i] + x_points[i + 1]) / 2)

    return xs


def trim_start(reduced_depths, reduced_st, reduced_pixel_values):
    # temporary smoothing to help find initial peak
    temp_st = gf1d(reduced_st, 1)
    peaks, props = find_peaks(temp_st, width=(5, 70))
    # check if it found any peaks
    if len(peaks) != 0:
        # check if the peak is actually near the start...
        half = reduced_depths[int(len(reduced_depths) / 2.5)]
        if reduced_depths[peaks[0]] < half:
            # trim all before first peak
            first_peak = peaks[0]
            reduced_depths = reduced_depths[first_peak:]
            reduced_pixel_values = reduced_pixel_values[first_peak:]
            reduced_st = reduced_st[first_peak:]
    return reduced_depths, reduced_st, reduced_pixel_values


def trim_end(reduced_depths, reduced_st, reduced_pixel_values):
    # temporary smoothing to help find deep peak
    temp_st = gf1d(reduced_st, 1)
    peaks, props = find_peaks(temp_st, width=(10, 100))
    # trim all after last peak
    if len(peaks) != 0:
        # check if near end
        half = reduced_depths[int(len(reduced_depths) / 2.5)]
        if reduced_depths[peaks[0]] > half:
            last_peak = peaks[-1]
            reduced_depths = reduced_depths[:last_peak]
            reduced_pixel_values = reduced_pixel_values[:last_peak]
            reduced_st = reduced_st[:last_peak]
    return reduced_depths, reduced_st, reduced_pixel_values


def process_raw_video_data(dataset, threshold, smoothing_factor):
    """Takes raw data extracted from a slice thickness video and
    processes it.

    Args:
        dataset (np.array): data from the video, should be 3 by data length. Data in mm.
        threshold (float): Set LCP as depth when pixel value drops below this threshold.
        smoothing_factor (int): choose how much to smooth the outputted STs by.

    Returns:
        list: [reduced_depths, reduced_st, reduced_pixel_value, dead zone, LCP] - the processed video data.
    """
    # sort and convert to mm
    ind = np.argsort(dataset[0])
    depths = np.array(dataset[0])[ind]
    slice_thicknesses = np.array(dataset[1])[ind]

    # smooth the pixel value curve for better LCP determination
    pixel_values = gf1d(np.array(dataset[2])[ind], 3)

    # calculate low contrast penetration from pixel values
    index = None
    for i, number in enumerate(pixel_values):
        # start from half way through so nothing at start is picked up
        if i < len(pixel_values) / 2:
            continue
        if number < threshold:
            index = i
            break

    # if pixel value falls below threshold, that depths is the LCP. If it does not,
    # LCP is the deepest point recorded.
    if index is not None:
        LCP = depths[index - 1]
    else:
        LCP = depths[-1]

    # average slice thicknesses and peak heights recorded for the same depth.
    reduced_st = []
    reduced_pixel_values = []
    reduced_depths, indices = np.unique(depths, return_index=True)
    pairs = list(pairwise(indices))
    for pair in pairs:
        reduced_st.append(np.median(slice_thicknesses[pair[0] : pair[1]]))
        reduced_pixel_values.append(np.median(pixel_values[pair[0] : pair[1]]))
    # pairwise misses out the last section so add it back in.
    reduced_st.append(np.median(slice_thicknesses[pairs[-1][1] :]))
    reduced_pixel_values.append(np.median(pixel_values[pairs[-1][1] :]))

    # presmooth
    reduced_st = gf1d(reduced_st, smoothing_factor)

    # trim points before first maximum
    reduced_depths, reduced_st, reduced_pixel_values = trim_start(
        reduced_depths, reduced_st, reduced_pixel_values
    )

    # trim points after final maximum
    reduced_depths, reduced_st, reduced_pixel_values = trim_end(
        reduced_depths, reduced_st, reduced_pixel_values
    )

    # smooth
    # reduced_st = gf1d(reduced_st, smoothing_factor)

    dead_zone = depths[0]

    return reduced_depths, reduced_st, reduced_pixel_values, dead_zone, LCP


def extract_Ls(required_video_paths:list, pipe_diameters:np.ndarray, threshold:float, smoothing_factor:int, aug=0.0):
    """Takes in videos for a probe and finds the depth range for which a series of
    slice thicknesses are larger.
    """

    # check diameters go from small to big else reverse them.
    if pipe_diameters[0] > pipe_diameters[-1]:
        pipe_diameters = pipe_diameters[::-1]

    vid_arrays = []
    LCPs = []
    # load in slice thickness plot data
    for vid_path in required_video_paths:
        dataset = np.genfromtxt(vid_path, dtype=float, delimiter=",").T

        # convert to mm
        dataset[:-1] *= 10

        (
            reduced_depths,
            reduced_st,
            reduced_pixel_values,
            dead_zone,
            LCP,
        ) = process_raw_video_data(dataset, threshold, smoothing_factor)

        vid_arrays.append([reduced_depths, reduced_st + (aug * reduced_st)])
        LCPs.append([dead_zone, LCP])

    """
    The following section loops through all the videos and diameters to find L for each diameter
    from all the videos. It stores the results in a dictionary called L_dict where the keys are
    the diameters.
    """
    shallow_dict = {}
    deep_dict = {}

    for diameter in pipe_diameters:
        # dictionaries to store the intersection points for each diameter from all videos
        shallow_dict[diameter] = []
        deep_dict[diameter] = []
        for i, vid_data in enumerate(vid_arrays):

            # points where the slice thickness curve crosses that pipe diameter.
            depth_vals = solve_for_x(vid_data[0], vid_data[1], diameter)
            # print(LCPs[i])
            dead_zone = LCPs[i][0]
            LCP = LCPs[i][1]

            # set defaults
            deep = LCP
            shallow = dead_zone

            # do not intersect
            if len(depth_vals) == 0:
                curve_min = vid_data[1].min()
                # pipe diameter is too small to be seen so end iteration
                # without appending.
                if diameter < curve_min:
                    continue

            # take the minimum slice thickness value as the bound between
            # shallow and deep.
            boundary = vid_data[0][np.argmin(vid_data[1])]
            depth_vals = np.array(depth_vals)

            # partition into shallows and deeps
            shallows = depth_vals[depth_vals < boundary]
            deeps = depth_vals[depth_vals > boundary]

            # if there are any intersections save them.
            if len(shallows) != 0 and np.std(shallows) <= 5:
                shallow = np.average(shallows)
            if len(deeps) != 0 and np.std(deeps) <= 5:
                deep = np.average(deeps)

            shallow_dict[diameter].append(np.max((shallow, dead_zone)))
            deep_dict[diameter].append(np.min((deep, LCP)))

    L_dict = {}
    L_list = []
    final_diameters = []
    for diameter in pipe_diameters:
        # pipe not seen
        if len(deep_dict[diameter]) == 0 and len(shallow_dict[diameter]) == 0:
            # the pipe was not seen so set L to 0.
            L_dict[diameter] = 0
            L_list.append(0)
        else:
            # calculate L for that diameter by subtracting the shallowest sighting from the deepest.
            L = np.max(deep_dict[diameter]) - np.min(shallow_dict[diameter])
            L_dict[diameter] = L
            L_list.append(L)
        final_diameters.append(diameter)

    # append the LCP and infinity
    L_list.append(np.max(LCPs) - np.min(LCPs))
    final_diameters.append(np.inf)

    return L_dict, L_list, np.array(final_diameters)


def main():
    if sys.argv[1] == "ST":
        get_slice_thickness(sys.argv[2])
    elif sys.argv[1] == "L":
        extract_Ls(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "all":
        videos = [27,28,29,30,50,51,56,57,60,61,62,63,21,
                  22,23,24,72,73,74,75,25,26,31,32,35,36,
                  37,38,68,69,70,71,64,65,66,67,78,79,80,81]
        for video in videos:
            n = str(video).zfill(2)
            try:
                get_slice_thickness(n)
            except:
                print(f"Error with vid{video}")
                continue


if __name__ == "__main__":
    main()
