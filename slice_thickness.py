from videos import Video
import matplotlib.pyplot as plt
import numpy as np


PATH = "/home/david/Documents/uni/year-5/ultrasound/"


def plotter(x, y, title):
    plt.plot(x, y)
    plt.xlabel("Depth/cm")
    plt.ylabel("Slice thickness/cm")
    plt.title(title)
    # plt.ylim(0.2, 1.6)


def main():
    viddata = {"roi": [632, 111, 232, 552],
               "filepath": f"{PATH}videos/", "filenumber": "07"}
    vid = Video(viddata)

    vid.save_slice_thickness_data(5, f"{PATH}scripts/analysed/vid{viddata['filenumber']}.txt")


main()
