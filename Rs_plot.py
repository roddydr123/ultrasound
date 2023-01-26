import matplotlib.pyplot as plt
from scipy.stats import pearsonr



# plt.scatter([269.51, 242.77, 281.17, 273.81, 241.2, 248.91], [99, 105, 70, 94, 88, 99])   # old Rs
# plt.scatter([0.349, 0.277, 0.121, 0.33, 0.514, 0.371],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21]) # old Drs
# plt.scatter([94.08, 67.24, 34.12, 90.36, 163, 92.24], [120, 71, 52, 91, 124, 120], marker="x")    # old Lrs

# plt.scatter([302.4, 276.16, 317.64, 303.57, 253, 284], [99, 105, 70, 94, 88, 99])   # deep Rs
# plt.scatter([0.373, 0.299, 0.13, 0.35, 0.534, 0.407],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21]) # deep Drs
plt.scatter([112.78, 82.66, 41.34, 106.36, 135, 115.49], [120, 71, 52, 91, 124, 120], marker="x")    # deep Lrs

plt.xlim([0, 150])
plt.ylim([0, 150])
plt.xlabel("Lr from ST")
plt.ylabel("Lr from EPP")

print(pearsonr([112.78, 82.66, 41.34, 106.36, 135, 115.49], [120, 71, 52, 91, 124, 120]))

plt.show()