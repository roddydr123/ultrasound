import cv2
import numpy as np
from scipy.signal import find_peaks


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"


class Video():
    def __init__(self, viddata):
        self.filename = viddata["filename"]
        self.start_deep = viddata["start_deep"]
        self.total_depth_cm = viddata["total_depth_cm"]
        self.total_depth_pixels = viddata["total_depth_pixels"]
        self.roi = viddata["roi"]
        self.cap = cv2.VideoCapture(f"{PATH}/videos/{self.filename}")
        self.frame_count = int(self.cap.get(7))

    def select_ROI(self, frame_index=0, roi=None):
        if roi:
            self.roi = roi
        else:
            self.cap.set(1, frame_index)
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.namedWindow("select window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("select window", frame.shape[0], frame.shape[1])
            self.roi = cv2.selectROI("select window", frame)
        return self.roi
 
    def get_profile(self, frame):
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = self.roi
        roi = greyscale[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        profile = np.average(list(roi), 1)
        cv2.imwrite(PATH + "scripts/output.jpg", roi)
        return profile

    def get_bkgd(self):
        if self.start_deep is True:
            self.cap.set(1, self.frame_count - 10)
            ret, frame = self.cap.read()
            profile = self.get_profile(frame)
            end_bkgd = profile[-int(len(profile) / 2):]

            self.cap.set(1, 10)
            ret, frame = self.cap.read()
            profile = self.get_profile(frame)
            start_bkgd = profile[:int(len(profile) / 2)]

            bkgd = np.concatenate((start_bkgd, end_bkgd))
        else:
            self.cap.set(1, 10)
            ret, frame = self.cap.read()
            profile = self.get_profile(frame)
            start_bkgd = profile[-int(len(profile) / 2):]

            self.cap.set(1, self.frame_count - 10)
            ret, frame = self.cap.read()
            profile = self.get_profile(frame)
            end_bkgd = profile[:int(len(profile) / 2)]

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
        x = np.linspace(0, len(profile), len(profile))
        # plt.plot(x, profile)
        peak, props = find_peaks(profile, distance=len(profile), width=(5, 70))
        width_cm = 0
        peak_depth_cm = 0
        if len(props['widths']) != 0:
            width = props['widths'][0]

            peak_depth_pixels = peak[0]
            conv_factor = self.total_depth_cm / self.total_depth_pixels
            peak_depth_cm = conv_factor * peak_depth_pixels

            width_cm = conv_factor * width

        return width_cm, peak_depth_cm

    def find_ROI(self):
        """
        Opens a window with a frame from the video, a rectangle can be
        dragged over the ROI. The row/column of the top left points is
        returned along with the width of the rectangle.
        """

        self.cap.set(1, 100)
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("select window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("select window", frame.shape[0], frame.shape[1])
        r = cv2.selectROI("select window", frame)

        return r

    def reverse(self):
        frame_list = []
        for i in range(self.total_frames):
            self.cap.set(1, self.total_frames - i)
            ret, frame = self.cap.read()
            frame_list.append(frame)

    def get_slice_thickness_data(self, resolution):
        widths = []
        depths = []
        width_indices = np.arange(0, self.frame_count, resolution)
        for index in width_indices:
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
