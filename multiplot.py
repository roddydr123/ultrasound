import matplotlib.pyplot as plt
import numpy as np


PATH = "/home/david/Documents/uni/year-5/ultrasound/"


def plotter(data, title):
    for dataset in data:
        plt.plot(dataset[0], dataset[1])
    plt.xlabel("Depth/cm")
    plt.ylabel("Slice thickness/cm")
    plt.title(title)
    plt.show()


def main():
    files_to_plot = ["vid07-highres"]
    data = []
    for file in files_to_plot:
        with open(f"{PATH}{file}.txt") as file:
            dat = np.genfromtxt(file, delimiter=",").T.tolist()
            data.append(dat)

    plotter(data, f"{files_to_plot}")


main()