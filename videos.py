import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"


def getFrames():
    cap = cv2.VideoCapture(PATH + "videos/vid06.mp4")
    total_frames = int(cap.get(7))

    widths = []

    for i in range(1, total_frames):
        if i % 200 == 0:
            cap.set(1, i)
            ret, frame = cap.read()
            width = analyseFrame(frame)
            widths.append(width)
    
    # x = np.linspace(0, len(widths), len(widths))

    # plt.plot(x, widths)
    plt.show()


def analyseFrame(frame):
    # print(frame.shape)
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = greyscale[100:700, 690:810]
    # cv2.imwrite(PATH + "scripts/output.jpg", roi)
    profile = np.average(list(roi), axis=1)
    x = np.linspace(0, len(profile), len(profile))
    plt.plot(x, profile)
    peak, props = find_peaks(profile, prominence=10, width=10)
    width = 0
    print(len(props['widths']))
    if len(props['widths']) != 0:
        width = props['widths'][0]
    return width


getFrames()
