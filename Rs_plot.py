import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


fig, ax = plt.subplots(figsize=(10, 10))

dat = np.loadtxt("analysed/gen3/allRs.txt", delimiter="\t", skiprows=1, dtype=str)

moran_dat = np.loadtxt("other_data/moran.txt", delimiter="\t", skiprows=1, dtype=str)

for probe in dat:
    if probe[1] != "EPP":
        x = float(probe[4])
        y = float(probe[3])
        ax.scatter(x, y)
        ax.annotate(f"{probe[0]}, {probe[1]}", (x, y))

for probe in moran_dat:
    x = float(probe[4])
    y = float(probe[3])
    ax.scatter(x, y)
    ax.annotate(f"{probe[0]}", (x, y))

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

xs = np.linspace(0, ax.get_xlim()[1], 100)
ys25 = 25 * xs
ys50 = 50 * xs
ys75 = 75 * xs
ax.plot(xs, ys25, "k--")
ax.annotate("R = 25", (xs[-1], ys25[-1]))
ax.plot(xs, ys50, "k--")
ax.annotate("R = 50", (xs[-50], ys50[-50]))
ax.plot(xs, ys75, "k--")
ax.annotate("R = 75", (xs[-70], ys75[-70]))

ax.set_xlabel("Characteristic resolution (mm)")
ax.set_ylabel("Depth of field (mm)")

# print(pearsonr(data[0], data[1]))
plt.show()
