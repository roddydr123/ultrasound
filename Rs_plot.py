import matplotlib.pyplot as plt
from scipy.stats import pearsonr



# plt.scatter([269.51, 242.77, 281.17, 273.81, 241.2, 248.91], [99, 105, 70, 94, 88, 99])
plt.scatter([0.349, 0.277, 0.121, 0.33, 0.514, 0.371],[1.21, 0.67, 0.75, 0.96, 1.97, 1.21])
# plt.scatter([94.08, 67.24, 34.12, 90.36, 163, 92.24], [120, 71, 52, 91, 124, 120], marker="x")
plt.xlim([0, 2.1])
plt.ylim([0, 2.1])
plt.xlabel("Dr from ST")
plt.ylabel("Dr from EPP")

# print(pearsonr([94.08, 67.24, 34.12, 90.36, 163, 92.24], [120, 71, 52, 91, 124, 120]))

plt.show()