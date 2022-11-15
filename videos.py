import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"

 
def get_profile(frame):
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = greyscale[114:700, 690:810]   # height, width
    profile = np.average(list(roi), 1)
    # cv2.imwrite(PATH + "scripts/output.jpg", roi)
    return profile


def get_bkgd(cap, total_frames, start_deep):
    if start_deep is True:
        cap.set(1, total_frames - 10)
        ret, frame = cap.read()
        profile = get_profile(frame)
        end_bkgd = profile[-int(len(profile) / 2):]

        cap.set(1, 10)
        ret, frame = cap.read()
        profile = get_profile(frame)
        start_bkgd = profile[:int(len(profile) / 2)]

        bkgd = np.concatenate((start_bkgd, end_bkgd))
    else:
        cap.set(1, 10)
        ret, frame = cap.read()
        profile = get_profile(frame)
        start_bkgd = profile[-int(len(profile) / 2):]

        cap.set(1, total_frames - 10)
        ret, frame = cap.read()
        profile = get_profile(frame)
        end_bkgd = profile[:int(len(profile) / 2)]

        bkgd = np.concatenate((start_bkgd, end_bkgd))

    if np.average(bkgd) < 2:
        bkgd = np.zeros_like(bkgd)

    return bkgd


def analyseFrame(frame, bkgd):
    profile = get_profile(frame)
    profile -= bkgd
    x = np.linspace(0, len(profile), len(profile))
    # plt.plot(x, profile)
    peak, props = find_peaks(profile, distance=len(profile), width=0)
    width = 0
    # print(f"props: {props}")
    # print(f"peaks: {peak}\n")
    if len(props['widths']) != 0:
        width = props['widths'][0]
    return width


def getFrames():
    start_deep = False
    cap = cv2.VideoCapture(PATH + "videos/vid04.mp4")
    total_frames = int(cap.get(7))

    bkgd = get_bkgd(cap, total_frames, start_deep)

    widths = []

    for i in range(1, total_frames):
        if i == 0:
            cap.set(1, i)
            ret, frame = cap.read()
            width = analyseFrame(frame, bkgd)
            widths.append(width)
    
    x = np.linspace(0, len(widths), len(widths))
    # x = np.linspace(0, len(bkgd), len(bkgd))

    # plt.plot(x, bkgd)
    plt.plot(x, widths)
    plt.show()

getFrames()
