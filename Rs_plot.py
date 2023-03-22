import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np
import matplotlib
from auto_res_check import calc_R
from slice_thickness import extract_Ls

plt.style.use("thesis.mplstyle")


def linear(x_array, m, c):
    return (m * x_array) + c


def profiles():
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot()

    colors = ["C0", "k"]
    names = np.arange(200, 1400, 200)
    for i, file in enumerate(names):
        datai = np.loadtxt(f"analysed/width_expt/30-{file}.txt").T
        x = datai[0]
        x *=10
        y = datai[1]
        color = colors[i % 2]
        ax.plot(x, y, color)
        peaks, properties = find_peaks(y, distance=len(x), width=(5, 70), height=(0, 5000))

        # ax.annotate(f"{np.round(properties['widths'], 1)} mm", [x[peaks], y[peaks] + 3])

        ax.plot(x[peaks], y[peaks], "x")

        ax.vlines(x=x[peaks], ymin=y[peaks] - properties["prominences"], ymax = y[peaks], color = "C1")

        ax.hlines(y=properties["width_heights"], xmin=x[int(properties["left_ips"])], xmax=x[int(properties["right_ips"])], color = "C1")
        

    names = np.arange(400, 1600, 200)
    for i, file in enumerate(names):
        data = np.loadtxt(f"analysed/width_expt/28-{file}.txt").T
        x = data[0]
        x *= 10
        y = data[1]
        color = colors[i % 2]
        ax.plot(x, y, color)
        peaks, properties = find_peaks(y, distance=len(x), width=(5, 70), height=(0, 5000))

        # ax.annotate(f"{np.round(properties['widths'], 1)} mm", [x[peaks], y[peaks] + 3])

        ax.plot(x[peaks], y[peaks], "x")

        ax.vlines(x=x[peaks], ymin=y[peaks] - properties["prominences"], ymax = y[peaks], color = "C1")

        ax.hlines(y=properties["width_heights"], xmin=x[int(properties["left_ips"])], xmax=x[int(properties["right_ips"])], color = "C1")

    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Pixel value")
    # ax.legend()
    ax.set_xlim(0, datai[0].max())
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    plt.tight_layout()
    plt.show()


def L_alpha():
    videos = [
    [27,28,29,30],
    [50,51,56,57],
    [60,61,62,63], 
    [21,22,23,24],
    [72,73,74,75], 
    [25,26,31,32],
    [35,36,37,38],
    [68,69,70,71],
    [64,65,66,67],
    [78,79,80,81]
    ]

    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters[::-1]
    L_dict, lengths, diameters = extract_Ls(videos[0], pipe_diameters, 20, 3)

    # lengths = np.array([0.0, 2.9, 26.0, 40.3, 55.0, 71.0, 81.3, 89.7, 93.4, 94.6, 97.2, 98.6])  # NHS data for ML6-15 31/03/15
    # lengths = np.array([0.0,0.0,3.8,18.1,23.9,70.1,85.1,128.6,147.0,165.6,165.1, 192.0]) # NHS data for 9LD 01/04/15
    # lengths = np.array([0.0,0.0,28.2,44.0,53.2,73.0,85.2,91.9,95.5,96.2,96.2,96.2])   # NHS data ML6-15 20/08/13
    # lengths = np.array([0.0, 2.0, 10.8, 22.7, 39.6, 44.9, 57.0, 65.8, 69.3, 72.5, 75.8, 76.8]) # NHS data 14L5 21/07/10
    # lengths = np.array([0.0, 0.0, 6.4, 24.4, 43.5, 58.5, 83.9, 97.7, 132.6, 137.9, 137.9, 137.9])  # NHS data 9L4 lab folder
    # lengths = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 41.0, 73.6, 121.0, 169.5, 197.1, 203.1, 221.5])   # NHS data C1-5 lab folder
    # lengths = [0,2.2,12.8,19.1,28.7,39.7,60.9,64.9,65.5,66.6,65.9,65.2] # NHS data in folder 18L6
    # lengths = [0,0,0,0,0,4.6,47.1,117.2,175.6,195,200.5,215.8]  # NHS data in folder 6C1
    # lengths = [0, 0, 0, 0, 0, 22.9, 72.6, 108, 148, 197, 201, 233]  # 4C1 NHS folder


    """Pipe/slice thickness diameters in mm"""
    # diameters = np.array([0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])
    # diameters = np.array([0.35, 0.42, 0.56, 0.70, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.9, np.inf])   # NHS data teams spreadsheet
    # diameters = np.array(
    #     [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
    # )  # NHS data lab folder

    # diameters = np.array(diameters)[::-1] / np.sqrt(
    #     np.cos(np.deg2rad(40))
    # )  # convert to effective diameter and reverse

    diameters = np.array(diameters)[::-1]
    inverse_diameters = 1 / diameters
    lengths = np.array(lengths)[::-1]

    Dr, Lr, R = calc_R(lengths, inverse_diameters, show=True)




def EPP_plot():
    data = np.loadtxt("analysed/gen3/all_data.txt", delimiter=",", dtype=str, skiprows=1)
    # format: probe name, EPP R, EPP Dr, EPP Lr,
    # R, R upper, R lower,
    # Dr, Dr upper, Dr lower,
    # Lr, Lr upper, Lr lower

    # to plot, R = 0, Dr = 1, Lr = 2
    index = 1

    code_uncs = [0.021, 0.04, 0.017]
    xlabels = ["Resolution Integral (EPP)", "Characteristic Resolution (EPP) (mm)", "Depth of Field (EPP) (mm)"]
    ylabels = ["Resolution Integral (ST)", "Characteristic Resolution (ST) (mm)", "Depth of Field (ST) (mm)"]

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
        offsets = [[xstandard - 9, ystandard + 10],[xstandard - 29, ystandard + 8],[xstandard + 3, ystandard - 4],[xstandard + 2, ystandard - 5],[xstandard + 2, ystandard - 2],
                [xstandard - 20, ystandard + 15],[xstandard - 42, ystandard - 4],[xstandard - 42, ystandard - 4],[xstandard - 7, ystandard + 15],[xstandard - 5, ystandard + 15]]

    arr = np.zeros((len(data), 12))
    names = []

    for i, line in enumerate(data):
        names.append(line[0])
        arr[i, :] = line[1:]

    x_array = arr[:, index]
    y_array = arr[:, (3 * index) + 3]
    xerr = x_array * 0.02   # 2% error
    yerr = np.array([arr[:, (3 * index) + 5], arr[:, (3 * index) + 4]])

    # 10% error for Lr
    if index == 2:
        yerr = 0.1 * y_array

    # combine with code uncertainty
    yerr = np.sqrt(yerr**2 + (code_uncs[index] * yerr)**2)

    # remove tiny error bars
    if index == 1:
        xerr[xerr < 0.02] = 0
    elif index == 2:
        xerr[xerr < 1.7] = 0

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()

    for name, xp, yp, offset in zip(names, x_array, y_array, offsets):
        ax.annotate(f"{name}", (xp + offset[0], yp + offset[1]), fontsize=12)

    ax.set_xlabel(xlabels[index])
    ax.set_ylabel(ylabels[index])

    for i, x, y, xe, ye in zip(range(len(x_array)), x_array, y_array, xerr, yerr):
        if i in [5,8,9]:
            if i == 5:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt="C1s", capsize=2, capthick=1, elinewidth=1, label="Curvilinear")
            else:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt="C1s", capsize=2, capthick=1, elinewidth=1)
            # curvi
            
        elif i in [0,6,7]:
            #linear
            if i == 0:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt="kx", capsize=2, capthick=1, elinewidth=1, label="Linear")
            else:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt="kx", capsize=2, capthick=1, elinewidth=1)
        else:
            if i == 4:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt="C0o", capsize=2, capthick=1, elinewidth=1, label="Multi-row")
            else:
                ax.errorbar(x, y, yerr=ye, xerr=xe, fmt="C0o", capsize=2, capthick=1, elinewidth=1)

    ax.legend()
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlim([0, ax.get_xlim()[1]])

    if index == 2:
        lim = max(ax.get_ylim()[1], ax.get_xlim()[1])
        ax.set_ylim([0, lim])
        ax.set_xlim([0, lim])
        popt, pcov = curve_fit(linear, x_array, y_array, sigma=yerr)
        long_x = np.linspace(0, ax.get_xlim()[1], 100)
        ax.plot(long_x, linear(long_x, *popt), linestyle="dashed")
        print(popt, np.sqrt(np.diag(pcov)))

    if index == 1:
        popt, pcov = curve_fit(linear, x_array, y_array, sigma = yerr[1])
        long_x = np.linspace(0, ax.get_xlim()[1], 100)
        ax.plot(long_x, linear(long_x, *popt), linestyle="dashed")
        print(popt, np.sqrt(np.diag(pcov)))

    print(round(pearsonr(x_array,y_array).statistic, 3))
    print(round(spearmanr(x_array,y_array).statistic, 3))

    # stop zeros overlapping
    if index == 1:
        offset = matplotlib.transforms.ScaledTranslation(0.05, 0, fig.dpi_scale_trans)
        ax.xaxis.get_majorticklabels()[0].set_transform(ax.xaxis.get_majorticklabels()[0].get_transform() + offset)

    plt.tight_layout()
    plt.show()


def R_plot():
    fig, ax = plt.subplots(figsize=(10, 7))

    # dat = np.loadtxt("analysed/gen3/reduced_Rs.txt", delimiter="\t", skiprows=1, dtype=str)
    dat = np.loadtxt("analysed/gen3/all_data.txt", delimiter=",", dtype=str)
    dat_xs = []
    dat_ys = []

    moran_dat = np.loadtxt("other_data/moran.txt", delimiter="\t", skiprows=1, dtype=str)
    mor_xs = []
    mor_ys = []

    inglis_dat = np.loadtxt("other_data/inglis.txt", delimiter="\t", skiprows=1, dtype=str)
    ing_xs = []
    ing_ys = []

    for probe in dat:
        x = float(probe[7])
        y = float(probe[10])
        dat_xs.append(x)
        dat_ys.append(y)
        ax.annotate(f"{probe[0]}", (x + 0.04, y))
    ax.scatter(dat_xs, dat_ys, label="Slice thickness")

    for probe in moran_dat:
        x = float(probe[4])
        y = float(probe[3])
        mor_xs.append(x)
        mor_ys.append(y)
        # ax.annotate(f"{probe[0]}", (x, y))
    ax.scatter(mor_xs, mor_ys, marker="x", label="Preclinical transducers (Moran et al.)")

    for probe in inglis_dat:
        x = float(probe[4])
        y = float(probe[3])
        ing_xs.append(x)
        ing_ys.append(y)
        # ax.annotate(f"{probe[0]}", (x + 0.04, y))
    ax.scatter(ing_xs, ing_ys, marker="s", label="Endocavity transducers (Inglis et al.)")

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

    ax.set_xlabel("$D_R$ (mm$^{-1}$)")
    ax.set_ylabel("$L_R$ (mm)")

    ax.legend()

    # print(pearsonr(data[0], data[1]))
    plt.show()


# R_plot()
EPP_plot()
# L_alpha()
# profiles()