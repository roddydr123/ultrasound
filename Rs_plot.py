import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


fig, ax = plt.subplots(figsize=(10, 10))

# ax.scatter([269.51, 242.77, 281.17, 273.81, 241.2, 248.91], [99, 105, 70, 94, 88, 99])   # old Rs
# ax.scatter([0.349, 0.277, 0.121, 0.33, 0.514, 0.371],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21]) # old Drs
# ax.scatter([94.08, 67.24, 34.12, 90.36, 163, 92.24], [120, 71, 52, 91, 124, 120], marker="x")    # old Lrs

# ax.scatter([302.4, 276.16, 317.64, 303.57, 253, 284], [99, 105, 70, 94, 88, 99])   # deep Rs
# ax.scatter([0.373, 0.299, 0.13, 0.35, 0.534, 0.407],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21]) # deep Drs
# ax.scatter([112.78, 82.66, 41.34, 106.36, 135, 115.49], [120, 71, 52, 91, 124, 120], marker="x")    # deep Lrs

# data = [3.37, 2.14, 1.3, 2.9, 4.7, 3.1, 1.77, 2.88, 1.33, 1.33, 2.34, 4.3], [1.21, 0.67, 0.75, 0.96, 1.97, 1.21, 0.76, 0.67, 0.75, 0.78, 0.96, 2.6]  # characteristic resolution mm
# data = [109, 72, 47, 90, 136, 98, 56.6, 76, 45, 45, 85.5, 149], [120, 71, 52, 91, 163, 120, 48, 71, 52, 51, 91, 179]  # depth of field mm
# data = [[32, 34, 36, 31, 29, 31.5, 32, 27, 34, 34, 36.59, 34.61], [99, 105, 70, 94, 88, 99, 64, 105, 70, 66, 94, 70]]  # resolution integral mm

# gen 3
data = [
    [2.956, 3.027, 3.062, 1.549, 1.53, 2.536, 2.435, 4.627, 1.652, 1.46, 5.346, 3.37],
    [0.96, 0.96, 0.67, 0.75, 0.75, 0.96, 0.96, 1.97, 0.779, 0.779, 2.269, 2.564],
]  # char res
# data = [[94.73, 99.12, 71.86, 38.74, 39.22, 86.76, 90.35, 134.01, 48.42, 41.1, 123.46, 118.72],[91, 91, 71, 52, 52, 91, 91, 163, 51.37, 51.37, 167.5, 179.41]]  # depth of field
# data = [[32.05, 32.75, 23.47, 25.02, 25.63, 34.21, 37.1, 28.96, 29.3, 28.16, 23.09, 35.23], [94, 94, 105, 70, 70, 94, 94, 88, 65.98, 65.98, 73.82, 69.97]] #res int mm

label = ["Characteristic resolution", "(mm)"]
# label = ["Depth of field", "(mm)"]
# label = ["Resolution integral", ""]

dat = np.loadtxt("analysed/gen3/allRs.txt", delimiter="\t", skiprows=1, dtype=str)

for probe in dat:
    x = float(probe[4])
    y = float(probe[3])
    ax.scatter(x, y)
    ax.annotate(f"{probe[0]}, {probe[1]}", (x, y))

ax.set_xlim([0, ax.get_xlim()[1]])
ax.set_ylim([0, ax.get_ylim()[1]])

# print(dat)
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
# ax.scatter(data[0], data[1])

# limit = np.max(np.array([ax.get_ylim(), ax.get_xlim()]))
# ax.legend()

ax.set_xlabel("Characteristic resolution (mm)")
ax.set_ylabel("Depth of field (mm)")

# print(pearsonr(data[0], data[1]))
plt.show()
