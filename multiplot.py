import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.ndimage import gaussian_filter1d as gf1d
from videos import fetch_video_details
from slice_thickness import process_raw_video_data


PATH = "/home/david/Documents/uni/year-5/ultrasound/scripts/analysed/gen2/"
PATH_TO_DETAILS = "/home/david/Documents/uni/year-5/ultrasound/videos/"


def plotter(data, title, details_list):
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, dataset in enumerate(data):
        # order data to be deeper monotonically.
        ind = np.argsort(dataset[0])
        x = np.array(dataset[0])[ind] * 10
        y = np.array(dataset[1])[ind] * 10
        h = np.array(dataset[2])[ind]
        x, y, h, dz, LCP = process_raw_video_data([x,y,h], 20, 3)
        # smooth the line and plot it.
        ax.plot(x, y, label=details_list[i])
        # ax.plot(x, gf1d(list(map(lambda x: x * 10, y)), 4), label=details_list[i])
        newax = ax.twinx()
        newax.plot(x, h, "r")
    ax.set_xlabel("Depth/cm")
    ax.set_ylabel("Slice thickness/mm")
    ax.legend()
    xlims = ax.get_xlim()
    ax.set_xlim(0, xlims[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    # newax.set_ylim(0, newax.get_ylim()[1])
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage multiplot.py vidXX vidYY...")
        sys.exit(1)
    files_to_plot = sys.argv[1:]
    data = []
    details_list = []
    for filename in files_to_plot:
        filepath = f"{PATH}{filename}.txt"
        with open(filepath) as file:
            dat = np.genfromtxt(file, delimiter=",").T.tolist()
            data.append(dat)
            try:
                details = fetch_video_details(PATH_TO_DETAILS, filename[-2:])
            except FileNotFoundError:
                details = filename
            details_list.append(details)
    plotter(data, files_to_plot, details_list)


main()