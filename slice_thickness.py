from videos import Video
import matplotlib.pyplot as plt
import numpy as np


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/"


def plotter(x, y):
    plt.plot(x, y)
    plt.xlabel("Depth/cm")
    plt.ylabel("Slice thickness/cm")
    # plt.ylim(0.2, 1.6)





def main():
    viddata = {"filename": "vid07.mp4", "start_deep": True,
               "total_depth_cm": 12, "roi": [632, 111, 232, 552],
               "filepath": f"{PATH}/videos/"}
    vid = Video(viddata)
    bkgd = vid.get_bkgd()
    # print(vid.read_info(f"{PATH}/videos/details.txt"))
    widths, depths = vid.get_slice_thickness_data(25)

    plotter(depths, widths)
    plt.show()


main()
