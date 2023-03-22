import matplotlib.pyplot as plt
import numpy as np
import sys
from videos import fetch_video_details
from slice_thickness import process_raw_video_data
import pathlib

plt.style.use("thesis.mplstyle")


p = pathlib.Path(__file__).parents
parentpath = p[1]

PATH = f"{parentpath}/scripts/analysed/gen3/"
PATH_TO_DETAILS = f"{parentpath}/videos/"

colors = ["C1", "k"]
labels = ["Un-processed", "Processed"]


def plotter(data, title, details_list):
    fig, ax = plt.subplots(figsize=(10, 4))
    # newax = ax.twinx()
    for i, dataset in enumerate(data):
        # order data to be deeper monotonically.
        ind = np.argsort(dataset[0])
        x = np.array(dataset[0])[ind] * 10
        y = np.array(dataset[1])[ind] * 10
        h = np.array(dataset[2])[ind]
        x, y, h, dz, LCP = process_raw_video_data([x, y, h], 20, 3)
        # smooth the line and plot it.
        ax.plot(x[x<=LCP], y[x<=LCP], linewidth=2)#, label=labels[i], color=colors[i])
        print(dz, LCP)
        # newax.plot(x, h, "r")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Slice Thickness (mm)")
    # ax.legend()
    xlims = ax.get_xlim()
    ax.set_xlim(0, xlims[1])
    # ax.hlines([0.5], [0], [xlims[1]], colors=["k"], linestyles=["dotted"])
    # ax.hlines([1.2, 1.2, 1.2], [0, 7.5, 17.5], [7.5, 17.5, xlims[1]], colors=["k", "k", "k"], linestyles=["dotted", "solid", "dotted"])
    # ax.hlines([2, 2, 2], [0, 2.52, 27.9], [2.52, 27.9, xlims[1]], colors=["k", "k", "k"], linestyles=["dotted", "solid", "dotted"])
    # ax.hlines([3, 3, 3], [0, 2.52, 42], [2.52, 42, xlims[1]], colors=["k", "k", "k"], linestyles=["dotted", "solid", "dotted"])
    # ax.hlines([4, 4, 4], [0, 2.52, 48.7], [2.52, 48.7, xlims[1]], colors=["k", "k", "k"], linestyles=["dotted", "solid", "dotted"])
    ax.set_ylim(0, ax.get_ylim()[1])
    # newax.set_ylim(0, newax.get_ylim()[1])
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    plt.tight_layout()
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
