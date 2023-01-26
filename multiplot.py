import matplotlib.pyplot as plt
import numpy as np
import sys


PATH = "/home/david/Documents/uni/year-5/ultrasound/scripts/analysed/"


def plotter(data, title):
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, dataset in enumerate(data):
        ax.plot(dataset[0], list(map(lambda x: x * 10, dataset[1])), label=title[i])
    ax.set_xlabel("Depth/cm")
    ax.set_ylabel("Slice thickness/mm")
    ax.legend(title)
    # ax.vlines([10.2], [0], [0.8])
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage multiplot.py vidXX vidYY...")
        sys.exit(1)
    files_to_plot = sys.argv[1:]
    data = []
    for file in files_to_plot:
        with open(f"{PATH}{file}.txt") as file:
            dat = np.genfromtxt(file, delimiter=",").T.tolist()
            data.append(dat)
    plotter(data, files_to_plot)


main()