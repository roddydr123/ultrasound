import cv2
import numpy as np
from scipy.signal import find_peaks
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt


class Video:
    def __init__(self, viddata):
        self.filenumber = viddata["filenumber"]
        self.filename = viddata["filename"]
        print(f"Analysing {self.filename}")
        self.filepath = viddata["filepath"]
        self.start_deep = True
        deets = fetch_video_details(self.filepath, self.filenumber)
        self.total_depth_cm = deets["total_depth"]
        self.cap = cv2.VideoCapture(self.filepath + "/" + self.filename)
        self.roi = deets["ROI"]
        self.total_depth_pixels = self.roi[3]
        self.frame_count = int(self.cap.get(7))
        self.get_bkgd()

    def get_profile(self, frame):
        """
        Extracts the pixel data from the ROI and flattens it to make a
        slice thickness graph.
        """
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = self.roi
        roi = greyscale[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
        profile = np.average(list(roi), 1)
        return profile

    def get_bkgd(self):
        self.cap.set(1, self.frame_count - 10)
        ret, frame = self.cap.read()
        profile = self.get_profile(frame)
        end_bkgd = profile[-int(len(profile) / 2) :]

        self.cap.set(1, 10)
        ret, frame = self.cap.read()
        profile = self.get_profile(frame)
        start_bkgd = profile[: int(len(profile) / 2)]

        bkgd = np.concatenate((start_bkgd, end_bkgd))

        if np.average(bkgd) < 2:
            # to avoid joining errors between the two bkgds
            bkgd = np.zeros_like(bkgd)

        if len(bkgd) < len(profile):
            # when the profile is split, int()ing it can mean it loses
            # an index. This re-adds it back in.
            bkgd = np.insert(bkgd, -1, 0)

        self.bkgd = bkgd
        return bkgd

    def analyseFrame(self, index):
        self.cap.set(1, index)
        ret, frame = self.cap.read()
        # skip frames which are not read in properly.
        if frame is None:
            return 0, 0, 0
        profile = self.get_profile(frame)
        profile -= self.bkgd

        profile[profile < 0] = 0

        # exclude very start and very end from peak finder since these often have reflections.
        # works by cutting 20 pixels from the top and bottom of the ROI before finding peaks, then
        # add 20 to all the peak locations.
        pixels_to_trim = 20
        trimmed_profile = profile[pixels_to_trim:-pixels_to_trim]
        pixels_to_trim += 1

        # get the height of the peak
        peak, props = find_peaks(
            trimmed_profile, distance=len(profile), width=(5, 70), height=(0, 5000)
        )
        height_list = props["peak_heights"]
        width_list = props["widths"]

        # if the peak finder hasn't been able to identify a peak or finds too many.
        if len(height_list) != 1 or len(width_list) != 1:
            return 0, 0, 0

        peak_height = height_list[0]
        width = width_list[0]

        # add back in the pixels we trimmed off to maintain accurate depth.
        peak_depth_pixels = peak[0] + pixels_to_trim
        conv_factor = self.total_depth_cm / self.total_depth_pixels
        peak_depth_cm = conv_factor * peak_depth_pixels

        width_cm = conv_factor * width

        # speed of sound conversion factor
        scf = 0.94
        width_cm *= scf
        peak_depth_cm *= scf

        # if index % 200 == 0:
        #     x = np.linspace(0, self.total_depth_cm * scf, len(profile))
        #     tosave = np.array([x, profile]).T
        #     np.savetxt(f"analysed/width_expt/27-{index}.txt", tosave)
        #     x = np.linspace(0, self.total_depth_cm * scf, len(profile))
        #     plt.plot(x, profile)

        #     plt.plot(peak_depth_cm, peak_height, "x")

        #     plt.vlines(x=peak_depth_cm, ymin=peak_height - props["prominences"], ymax = peak_height, color = "C1")

        #     plt.hlines(y=props["width_heights"], xmin=(props["left_ips"] + pixels_to_trim) * conv_factor * scf,
        #                xmax=(props["right_ips"] + pixels_to_trim) * conv_factor * scf, color = "C1")
        #     plt.show()

        return width_cm, peak_depth_cm, peak_height

    def get_slice_thickness_data(self, resolution):
        widths = []
        depths = []
        peak_heights = []
        width_indices = np.arange(0, self.frame_count, resolution)
        for index in tqdm(width_indices):
            width, depth_cm, peak_height = self.analyseFrame(index)
            # check if the peak was found successfully and skip if not
            if width == 0 or peak_height == 0:
                continue
            widths.append(width)
            depths.append(depth_cm)
            peak_heights.append(peak_height)
        widths, depths, heights = self.trim_overlaps(widths, depths, peak_heights)
        return widths, depths, heights

    def trim_overlaps(self, widths, depths, heights):
        width_diffs = np.sqrt(np.diff(widths) ** 2)
        threshold = max(widths) / 5
        bad_indices = []
        for i, diff in enumerate(width_diffs):
            if diff >= threshold:
                bad_indices.append(i)
        depths = np.array(depths)
        widths = np.array(widths)
        heights = np.array(heights)
        mask = np.ones(len(widths), dtype=bool)
        mask[bad_indices] = False
        depths = depths[mask]
        widths = widths[mask]
        heights = heights[mask]

        depth_diffs = np.diff(depths)
        bads = [0]
        for i, diff in enumerate(depth_diffs):
            if diff > 0:
                bads.append(i)
        mask2 = np.ones(len(depths), dtype=bool)
        mask2[bads] = False
        depths = depths[mask2]
        widths = widths[mask2]
        heights = heights[mask2]
        return widths, depths, heights

    def save_slice_thickness_data(self, resolution, filepath):
        print(f"--> {filepath}")
        widths, depths, heights = self.get_slice_thickness_data(resolution)

        data = np.array([depths, widths, heights]).T.tolist()

        with open(filepath, "w+") as file:
            for i, line in enumerate(data):
                file.write(f"{line[0]},{line[1]},{line[2]}\n")


def fetch_video_details(filepath, filenumber):
    """Retrieve the depth from details.txt"""
    with open(f"{filepath}/details.txt", "r") as file:
        for line in csv.reader(file, delimiter="\t"):
            if line[0] == filenumber:
                tdepth = float(line[2])

                ROI_tuple = line[5].split(sep=",")
                ROI_list = list(map(int, ROI_tuple))

                fdepth_tuple = line[4].split(sep=",")
                fdepth_list = list(map(float, fdepth_tuple))

                probe = line[1]

                freq = line[3]
                break
    return {
        "video_number": filenumber,
        "total_depth": tdepth,
        "ROI": ROI_list,
        "focus_depth": fdepth_list,
        "probe": probe,
        "frequency": freq,
    }
