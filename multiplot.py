import matplotlib.pyplot as plt


PATH = "/mnt/c/users/david/Documents/uni/year-5/ultrasound/scripts/analysed/"


def plotter(data, title):
    for dataset in data:
        plt.plot(dataset[0], dataset[1])
    plt.xlabel("Depth/cm")
    plt.ylabel("Slice thickness/cm")
    plt.title(title)


def main():
    files_to_plot = ["vid06", "vid10"]
    data = []
    for file in files_to_plot:
        with open(f"{PATH}{file}.txt") as file:
            dat = np.genfromtxt(file, delimiter=",").T.tolist()
            data.append(dat)

    plotter(data, f"{files_to_plot}")


main()