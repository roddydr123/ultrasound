import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

plt.style.use("thesis.mplstyle")


fig, ax = plt.subplots(figsize=(10, 10))

dat = np.loadtxt("analysed/gen3/allRs.txt", delimiter="\t", skiprows=1, dtype=str)
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
        ax.annotate(f"{probe[0]}", (x + 0.04, y))
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
ys75 = 75 * xs
ax.plot(xs, ys25, "k--")
ax.annotate("R = 25", (xs[-20] + 0.1, ys25[-20]), fontsize="large")
ax.plot(xs, ys50, "k-")
ax.annotate("R = 50", (xs[-55] + 0.07, ys50[-55]), fontsize="large")
ax.plot(xs, ys75, "k--")
ax.annotate("R = 75", (xs[-70] + 0.05, ys75[-70]), fontsize="large")

ax.set_xlabel("Characteristic resolution (mm)")
ax.set_ylabel("Depth of field (mm)")

ax.legend()

# print(pearsonr(data[0], data[1]))
plt.show()
