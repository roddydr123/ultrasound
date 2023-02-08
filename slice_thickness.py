from videos import Video
import numpy as np
# from scipy.interpolate import UnivariateSpline as us
import matplotlib.pyplot as plt
from itertools import pairwise
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.signal import find_peaks
import sys


PATH = "/home/david/Documents/uni/year-5/ultrasound/"


def get_slice_thickness(number):
    viddata = {
               "filepath": f"{PATH}videos/", "filenumber": f"{number}"}
    vid = Video(viddata)

    vid.save_slice_thickness_data(15, f"{PATH}scripts/analysed/gen2/vid{viddata['filenumber']}.txt")




def solve_for_x(x_points, y_points, y):

    xs = []

    # Find the interval containing the solution
    for i in range(len(x_points) - 1):
        if y_points[i] <= y <= y_points[i + 1] or y_points[i] >= y >= y_points[i + 1]:
            xs.append((x_points[i] + x_points[i + 1]) / 2)
    
    return xs



def trim_start(reduced_depths, reduced_st):
        # temporary smoothing to help find initial peak
        temp_st = gf1d(reduced_st, 1)
        peaks, props = find_peaks(temp_st, width=(5, 70))
        # trim all before first peak
        first_peak = peaks[0]
        reduced_depths = reduced_depths[first_peak:]
        reduced_st = reduced_st[first_peak:]
        return reduced_depths, reduced_st



def trim_end(reduced_depths, reduced_st):
    # temporary smoothing to help find initial peak
    temp_st = gf1d(reduced_st, 1)
    peaks, props = find_peaks(temp_st, width=(10, 100))
    # trim all after last peak
    last_peak = peaks[-1]
    reduced_depths = reduced_depths[:last_peak]
    reduced_st = reduced_st[:last_peak]
    return reduced_depths, reduced_st



def extract_Ls(required_videos, pipe_diameters, smoothing_factor=3, threshold=20):
    """Takes in videos for a probe and finds the depth range for which a series of
    slice thicknesses are larger.
    """

    # required_videos = [45, 54]
    # range of slice thicknesses to find L for.
    # pipe_diameters = [1,4,5,5.9,6] # np.linspace(2, 8, 1)

    vid_arrays = []
    LCPs = []
    # load in slice thickness plot data
    for vid_number in required_videos:
        vid_path = f"{PATH}scripts/analysed/vid{vid_number}.txt"
        dataset = np.genfromtxt(vid_path, dtype=float, delimiter=",").T

        # sort and convert to mm
        ind = np.argsort(dataset[0])
        depths = np.array(dataset[0])[ind] * 10
        slice_thicknesses = np.array(dataset[1])[ind] * 10
        pixel_values = np.array(dataset[2])[ind]

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

        # average slice thicknesses recorded for the same depth.
        reduced_st = []
        reduced_depths, indices = np.unique(depths, return_index=True)
        pairs = list(pairwise(indices))
        for pair in pairs:
            reduced_st.append(np.median(slice_thicknesses[pair[0]:pair[1]]))
        # pairwise misses out the last section so add it back in.
        reduced_st.append(np.median(slice_thicknesses[list(pairs)[-1][1]:]))

        # trim points before first maximum
        reduced_depths, reduced_st = trim_start(reduced_depths, reduced_st)

        # trim points after final maximum
        reduced_depths, reduced_st = trim_end(reduced_depths, reduced_st)

        # smooth
        reduced_st = gf1d(reduced_st, smoothing_factor)

        dead_zone = depths[0]

        vid_arrays.append([reduced_depths, reduced_st])
        LCPs.append([dead_zone, LCP])

        # print(dead_zone, LCP)

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

            # does not cross
            if len(depth_vals) == 0:
                curve_max = vid_data[1].max()
                # pipe diameter is too small to be seen.
                if diameter < curve_max:
                    # shallow_dict.pop(diameter)
                    # deep_dict.pop(diameter)
                    pass
                elif diameter > curve_max:
                    # pipe is definitely seen both deep and shallow, 
                    shallow_dict[diameter].append(LCPs[i][0])
                    deep_dict[diameter].append(LCPs[i][1])

            # intersects twice, once shallow and once deep
            elif len(depth_vals) == 2:
                shallow_dict[diameter].append(depth_vals[0])
                deep_dict[diameter].append(depth_vals[1])

            # intersects once, deep or shallow?
            elif len(depth_vals) == 1:
                # slightly more than halfway depth since curves are
                # usually skewed towards the start
                half = vid_data[0][int(len(vid_data[0]) / 2.5)]
                if depth_vals[0] > half:
                    # its a deep intersection so use dead zone for shallow
                    deep_dict[diameter].append(depth_vals[0])
                    shallow_dict[diameter].append(LCPs[i][0])
                else:
                    # its shallow so use LCP for deep
                    shallow_dict[diameter].append(depth_vals[0])
                    deep_dict[diameter].append(LCPs[i][1])
            
            else:
                print(f"curve intersects y={diameter} at x={depth_vals}")


    L_dict = {}
    for diameter in pipe_diameters:
        # pipe not seen
        if len(deep_dict[diameter]) == 0 and len(shallow_dict[diameter]) == 0:
            L_dict[diameter] = 0
        else:
            # calculate L for that diameter by subtracting the shallowest sighting from the deepest.
            L_dict[diameter] = np.max(deep_dict[diameter]) - np.min(shallow_dict[diameter])

    
    return L_dict
    


def main():
    if sys.argv[1] == "ST":
        get_slice_thickness(sys.argv[2])
    elif sys.argv[1] == "L":
        extract_Ls(sys.argv[2], sys.argv[3])


if __name__=="__main__":
    main()
