import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        # cv2.imwrite(PATH + "scripts/output.jpg", roi)
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
        

def main():
    viddata = {"filename": "vid01.mp4", "start_deep": False,
               "total_depth_cm": 20, "roi": [687, 106, 117, 524],
               "total_depth_pixels": 563}
    vid = Video(viddata)
    bkgd = vid.get_bkgd()
    widths = []
    depths = []
    width_indices = np.arange(0, vid.frame_count, 25)
    for index in width_indices:
        width, depth_cm = vid.analyseFrame(index, subtract_bkgd=True)
        if width != 0:
            widths.append(width)
            depths.append(depth_cm)

    # x = np.linspace(0, len(widths), len(widths))
    # x = vid.get_depth_cm()
    # x = np.linspace(0, len(bkgd), len(bkgd))

    # plt.plot(x, bkgd)
    plt.plot(depths, widths)
    plt.show()


main()
