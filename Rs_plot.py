import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import matplotlib

plt.style.use("thesis.mplstyle")


def profiles():
    neames = [400, 600, 800, 1000, 1200]
    for file in neames:
        data = np.loadtxt(f"analysed/width_expt/27-{file}.txt").T
        plt.plot(data[0], data[1])
    plt.show()



def EPP_plot():
    data = np.loadtxt("analysed/gen3/all_data.txt", delimiter=",", dtype=str)
    # format: probe name, EPP R, EPP Dr, EPP Lr,
    # R, R upper, R lower,
    # Dr, Dr upper, Dr lower,
    # Lr, Lr upper, Lr lower

    # to plot, R = 0, Dr = 1, Lr = 2
    index = 2

    code_uncs = [0.021, 0.04, 0.017]
    xlabels = ["R from EPP", "$D_R$ from EPP (mm$^{-1}$)", "$L_R$ from EPP (mm)"]
    ylabels = ["R from slice thickness", "$D_R$ from slice thickness (mm$^{-1}$)", "$L_R$ from slice thickness (mm)"]

    # for R plot
    if index == 0:
        ystandard = 0.8
        xstandard = 1
        offsets = [[xstandard, ystandard],[xstandard - 15, ystandard],[xstandard, ystandard],[xstandard - 12, ystandard],[xstandard, ystandard],
                [xstandard, ystandard],[xstandard - 20, ystandard - 1],[xstandard - 20, ystandard - 1],[xstandard, ystandard],[xstandard, ystandard]]
    elif index == 1:
        ystandard = 0.08
        xstandard = 0.02
        offsets = [[xstandard, ystandard],[xstandard, ystandard],[xstandard - 0.23, ystandard - 0.1],[xstandard - 0.28, ystandard - 0.1],[xstandard - 0.28, ystandard-0.2],
                [xstandard, ystandard],[xstandard + 0.02, ystandard - 0.1],[xstandard + 0.02, ystandard-0.1],[xstandard, ystandard],[xstandard - 0.17, ystandard]]
    elif index == 2:
        ystandard = 2
        xstandard = 1
        offsets = [[xstandard, ystandard],[xstandard - 26, ystandard],[xstandard, ystandard],[xstandard + 2, ystandard - 5],[xstandard + 2, ystandard - 2],
                [xstandard - 20, ystandard],[xstandard, ystandard],[xstandard - 34, ystandard - 3],[xstandard, ystandard - 9],[xstandard - 0.4, ystandard]]

    arr = np.zeros((len(data), 12))
    names = []

    for i, line in enumerate(data):
        names.append(line[0])
        arr[i, :] = line[1:]

    x = arr[:, index]
    y = arr[:, (3 * index) + 3]
    xerr = x * 0.02   # 2% error
    yerr = np.array([arr[:, (3 * index) + 5], arr[:, (3 * index) + 4]])

    # 10% error for Lr
    if index == 2:
        yerr = 0.1 * y

    # combine with code uncertainty
    yerr = np.sqrt(yerr**2 + (code_uncs[index] * yerr)**2)

    # remove tiny error bars
    if index == 1:
        xerr[xerr < 0.02] = 0
    elif index == 2:
        xerr[xerr < 1.7] = 0

    fig, ax = plt.subplots()

    for name, xp, yp, offset in zip(names, x, y, offsets):
        ax.annotate(f"{name}", (xp + offset[0], yp + offset[1]))
        # print(name, round(100 * yp/xp, 1))
    ax.set_xlabel(xlabels[index])
    ax.set_ylabel(ylabels[index])
    print(pearsonr(x,y))
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="kx", capsize=2, capthick=1, elinewidth=1)
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])

    # stop zeros overlapping
    if index == 1:
        offset = matplotlib.transforms.ScaledTranslation(0.05, 0, fig.dpi_scale_trans)
        ax.xaxis.get_majorticklabels()[0].set_transform(ax.xaxis.get_majorticklabels()[0].get_transform() + offset)

    plt.tight_layout()
    plt.show()


def R_plot():
    fig, ax = plt.subplots(figsize=(10, 7))

    # dat = np.loadtxt("analysed/gen3/reduced_Rs.txt", delimiter="\t", skiprows=1, dtype=str)
    dat = np.loadtxt("analysed/gen3/auto_res_check.txt", delimiter=",", dtype=str)
    dat_xs = []
    dat_ys = []

    moran_dat = np.loadtxt("other_data/moran.txt", delimiter="\t", skiprows=1, dtype=str)
    mor_xs = []
    mor_ys = []

    inglis_dat = np.loadtxt("other_data/inglis.txt", delimiter="\t", skiprows=1, dtype=str)
    ing_xs = []
    ing_ys = []

    for probe in dat:
        if probe[1] != "EPP":
            x = float(probe[4])
            y = float(probe[3])
            dat_xs.append(x)
            dat_ys.append(y)
            ax.annotate(f"{probe[0][1:-1]}", (x + 0.04, y))
    ax.scatter(dat_xs, dat_ys, label="Slice thickness measurements")

    for probe in moran_dat:
        x = float(probe[4])
        y = float(probe[3])
        mor_xs.append(x)
        mor_ys.append(y)
        # ax.annotate(f"{probe[0]}", (x, y))
    ax.scatter(mor_xs, mor_ys, marker="x", label="Moran et al.")

    for probe in inglis_dat:
        x = float(probe[4])
        y = float(probe[3])
        ing_xs.append(x)
        ing_ys.append(y)
        # ax.annotate(f"{probe[0]}", (x + 0.04, y))
    ax.scatter(ing_xs, ing_ys, marker="s", label="Inglis et al.")

    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])

    xs = np.linspace(0, ax.get_xlim()[1], 100)
    ys25 = 25 * xs
    ys50 = 50 * xs
    ys12 = 12.5 * xs
    ax.plot(xs, ys25, "k-")
    ax.annotate("R = 25", (xs[-20] + 0.1, ys25[-20]), fontsize="large")
    ax.plot(xs, ys50, "k--")
    ax.annotate("R = 50", (xs[-55] + 0.07, ys50[-55]), fontsize="large")
    ax.plot(xs, ys12, "k--")
    ax.annotate("R = 12.5", (xs[-20], ys12[-20]-3), fontsize="large")

    ax.set_xlabel("Characteristic resolution (mm)")
    ax.set_ylabel("Depth of field (mm)")

    ax.legend()

    # print(pearsonr(data[0], data[1]))
    plt.show()


# R_plot()
EPP_plot()
# profiles()