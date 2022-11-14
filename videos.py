import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"


def getFrames():
    cap = cv2.VideoCapture(PATH + "videos/vid08.mp4")
    total_frames = int(cap.get(7))

    widths = []
    all = np.array([0.0]*586)
    count = 0

    for i in range(1, total_frames):
        if i % 50 == 0:
            cap.set(1, i)
            ret, frame = cap.read()
            width, profile = analyseFrame(frame)
            widths.append(width)
            all += np.array(profile)
            count += 1

    all /= count
    x = np.linspace(0, 600, len(all))
    plt.plot(x, all)
    plt.ylim(0, 200)
    
    # x = np.linspace(0, len(widths), len(widths))

    # plt.plot(x, widths)
    plt.show()


def analyseFrame(frame):
    # print(frame.shape)
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = greyscale[114:700, 690:810]   # height, width
    # cv2.imwrite(PATH + "scripts/output.jpg", roi)
    profile = np.average(list(roi), 1)
    x = np.linspace(0, len(profile), len(profile))
    # plt.plot(x, profile)
    peak, props = find_peaks(profile, prominence=20, width=0)
    width = 0
    # print(f"props: {props}")
    # print(f"peaks: {peak}\n")
    # if len(props['widths']) != 0:
        # width = props['widths'][0]
    return width, profile


getFrames()
