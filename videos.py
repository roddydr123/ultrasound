import cv2
import numpy as np
from scipy.signal import find_peaks
import csv
from tqdm import tqdm


class Video():
    def __init__(self, viddata):
        self.filenumber = viddata["filenumber"]
        self.filename = f"vid{self.filenumber}.mp4"
        print(f"Analysing {self.filename}")
        self.filepath = viddata["filepath"]
        self.start_deep = True
        self.total_depth_cm, ROI_list = self.fetch_video_details()
        self.cap = cv2.VideoCapture(self.filepath + self.filename)
        try:
            self.roi = viddata["roi"]
        except KeyError:
            self.roi = ROI_list
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
        roi = greyscale[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        profile = np.average(list(roi), 1)
        # cv2.imwrite(PATH + "scripts/output.jpg", roi)
        return profile

    def get_bkgd(self):
        self.cap.set(1, self.frame_count - 10)
        ret, frame = self.cap.read()
        profile = self.get_profile(frame)
        end_bkgd = profile[-int(len(profile) / 2):]

        self.cap.set(1, 10)
        ret, frame = self.cap.read()
        profile = self.get_profile(frame)
        start_bkgd = profile[:int(len(profile) / 2)]

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

    def analyseFrame(self, index, subtract_bkgd=True):
        self.cap.set(1, index)
        ret, frame = self.cap.read()
        profile = self.get_profile(frame)
        if subtract_bkgd is True:
            profile -= self.bkgd

        # exclude very start and very end from peak finder since these often have reflections.
        # works by cutting 20 pixels from the top and bottom of the ROI before finding peaks, then
        # add 20 to all the peak locations.
        pixels_to_trim = 20
        trimmed_profile = profile[pixels_to_trim:-pixels_to_trim]

        peak, props = find_peaks(trimmed_profile, distance=len(profile), width=(5, 70))
        width_cm = 0
        peak_depth_cm = 0
        if len(props['widths']) != 0:
            width = props['widths'][0]

            # add back in the pixels we trimmed off to maintain accurate depth.
            peak_depth_pixels = peak[0] + pixels_to_trim
            conv_factor = self.total_depth_cm / self.total_depth_pixels
            peak_depth_cm = conv_factor * peak_depth_pixels

            width_cm = conv_factor * width

        return width_cm, peak_depth_cm

    def get_slice_thickness_data(self, resolution):
        widths = []
        depths = []
        width_indices = np.arange(0, self.frame_count, resolution)
        for index in tqdm(width_indices):
            width, depth_cm = self.analyseFrame(index, subtract_bkgd=True)
            if width != 0:
                widths.append(width)
                depths.append(depth_cm)
        widths, depths = self.trim_overlaps(widths, depths)
        return widths, depths

    def trim_overlaps(self, widths, depths):
        width_diffs = np.sqrt(np.diff(widths)**2)
        threshold = max(widths) / 5
        bad_indices = []
        for i, diff in enumerate(width_diffs):
            if diff >= threshold:
                bad_indices.append(i)
        depths = np.array(depths)
        widths = np.array(widths)
        mask = np.ones(len(widths), dtype=bool)
        mask[bad_indices] = False
        depths = depths[mask]
        widths = widths[mask]

        depth_diffs = np.diff(depths)
        bads = [0]
        for i, diff in enumerate(depth_diffs):
            if diff > 0:
                bads.append(i)
        mask2 = np.ones(len(depths), dtype=bool)
        mask2[bads] = False
        depths = depths[mask2]
        widths = widths[mask2]
        return widths, depths

    def save_slice_thickness_data(self, resolution, filepath):
        widths, depths = self.get_slice_thickness_data(resolution)

        data = np.array([depths, widths]).T.tolist()

        with open(filepath, "w+") as file:
            for line in data:
                file.write(f"{line[0]},{line[1]}\n")

    
    def fetch_video_details(self):
        """Retrieve the depth from details.txt"""
        with open(f"{self.filepath}/details.txt", "r") as file:
            for line in csv.reader(file, delimiter="\t"):
                if line[0] == self.filenumber:
                    tdepth = float(line[2])
                    ROI_tuple = line[5].split(sep=",")
                    ROI_list = list(map(int, ROI_tuple))
                    break
        return tdepth, ROI_list
