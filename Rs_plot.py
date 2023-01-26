import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


fig, ax = plt.subplots()

# ax.scatter([269.51, 242.77, 281.17, 273.81, 241.2, 248.91], [99, 105, 70, 94, 88, 99])   # old Rs
# ax.scatter([0.349, 0.277, 0.121, 0.33, 0.514, 0.371],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21]) # old Drs
# ax.scatter([94.08, 67.24, 34.12, 90.36, 163, 92.24], [120, 71, 52, 91, 124, 120], marker="x")    # old Lrs

# ax.scatter([302.4, 276.16, 317.64, 303.57, 253, 284], [99, 105, 70, 94, 88, 99])   # deep Rs
# ax.scatter([0.373, 0.299, 0.13, 0.35, 0.534, 0.407],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21]) # deep Drs
# ax.scatter([112.78, 82.66, 41.34, 106.36, 135, 115.49], [120, 71, 52, 91, 124, 120], marker="x")    # deep Lrs

# data = [3.37, 2.14, 1.3, 2.9, 4.7, 3.1, 1.77], [1.21, 0.67, 0.75, 0.96, 1.97, 1.21, 0.76]  # characteristic resolution mm
# data = [109, 72, 47, 90, 136, 98, 56.6], [120, 71, 52, 91, 163, 120, 48]  # depth of field mm
data = [[32, 34, 36, 31, 29, 31.5, 32], [99, 105, 70, 94, 88, 99, 64]]  # resolution integral mm

label = "R"

ax.scatter(data[0], data[1])

limit = np.max(np.array([ax.get_ylim(), ax.get_xlim()]))

ax.set_xlim([0, 1.1 * limit])
ax.set_ylim([0, 1.1 * limit])

ax.set_xlabel(f"{label} from ST")
ax.set_ylabel(f"{label} from EPP")

print(pearsonr(data[0], data[1]))
plt.show()