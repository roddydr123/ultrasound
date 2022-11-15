import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"


class Video():
    def __init__(self, filename, start_deep):
        self.filename = filename
        self.start_deep = start_deep
        self.cap = cv2.VideoCapture(f"{PATH}/videos/{filename}")
        self.frame_count = int(self.cap.get(7))
        self.bkgd = self.get_bkgd()
 
    def get_profile(self, frame):
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = greyscale[114:700, 690:810]   # height, width
        profile = np.average(list(roi), 1)
        # cv2.imwrite(PATH + "scripts/output.jpg", roi)
        return profile

    def get_bkgd(self):
        if self.start_deep is True:
            self.cap.set(1, self.frame_count - 10)
            ret, frame = self.cap.read()
            roi = cv2.selectROI("select image", frame)
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
            bkgd = np.zeros_like(bkgd)

        return bkgd

    def analyseFrame(self, index, subtract_bkgd=True):
        self.cap.set(1, index)
        ret, frame = self.cap.read()
        profile = self.get_profile(frame)
        if subtract_bkgd is True:
            profile -= self.bkgd
        # x = np.linspace(0, len(profile), len(profile))
        # plt.plot(x, profile)
        peak, props = find_peaks(profile, distance=len(profile), width=0)
        width = 0
        # print(f"props: {props}")
        # print(f"peaks: {peak}\n")
        if len(props['widths']) != 0:
            width = props['widths'][0]
        return width


def main():
    vid = Video("vid04.mp4", start_deep=False)
    widths = []
    width_indices = np.arange(0, vid.frame_count, 50)
    for index in width_indices:
        width = vid.analyseFrame(index, subtract_bkgd=True)
        widths.append(width)

    x = np.linspace(0, len(widths), len(widths))
    # x = np.linspace(0, len(bkgd), len(bkgd))

    # plt.plot(x, bkgd)
    plt.plot(x, widths)
    plt.show()


main()
