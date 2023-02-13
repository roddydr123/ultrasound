import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


fig, ax = plt.subplots(figsize=(10, 10))

# gen 3
data = [
    [2.956, 3.027, 3.062, 1.549, 1.53, 2.536, 2.435, 4.627, 1.652, 1.46, 5.346, 3.37],
    [0.96, 0.96, 0.67, 0.75, 0.75, 0.96, 0.96, 1.97, 0.779, 0.779, 2.269, 2.564],
]

dat = np.loadtxt("analysed/gen3/allRs.txt", delimiter="\t", skiprows=1, dtype=str)

for probe in dat:
    x = float(probe[4])
    y = float(probe[3])
    ax.scatter(x, y)
    ax.annotate(f"{probe[0]}, {probe[1]}", (x, y))

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
