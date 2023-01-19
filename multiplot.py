import matplotlib.pyplot as plt
import numpy as np
import sys


PATH = "/home/david/Documents/uni/year-5/ultrasound/scripts/analysed/"


def plotter(data, title):
    for i, dataset in enumerate(data):
        plt.plot(dataset[0], dataset[1], label=title[i])
    plt.xlabel("Depth/cm")
    plt.ylabel("Slice thickness/cm")
    plt.legend()
    plt.show()


def main():
    files_to_plot = sys.argv[1:]
    data = []
    for file in files_to_plot:
        with open(f"{PATH}{file}.txt") as file:
            dat = np.genfromtxt(file, delimiter=",").T.tolist()
            data.append(dat)

    plotter(data, files_to_plot)


main()