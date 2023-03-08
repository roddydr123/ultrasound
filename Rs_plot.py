import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

plt.style.use("thesis.mplstyle")



def EPP_plot():
    epp_dat = np.loadtxt("analysed/gen3/EPP_data.txt", delimiter="\t", dtype=str)
    dat = np.loadtxt("analysed/gen3/auto_res_check.txt", delimiter=",", dtype=str)

    epp_data_clean = []
    st_data_clean = []
    names = []
    
    for line in dat:
        p_type = line[0].strip()[1:-1]
        names.append(p_type)
        st_data_clean.append([x.strip() for x in line[2:]])
        for epp in epp_dat:
            if epp[0] == p_type:
                epp_data_clean.append(list(epp[2:]))

    epp_data_clean = np.array(epp_data_clean).T
    st_data_clean = np.array(st_data_clean).T
    x = epp_data_clean[0].astype(float)
    y = st_data_clean[0].astype(float)

    fig, ax = plt.subplots()
    for name, xp, yp in zip(names, x, y):
        ax.annotate(f"{name}", (xp + 0.02, yp))
        print(name, round(100 * yp/xp, 1))
    print(pearsonr(x,y))
    ax.scatter(x, y)
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
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