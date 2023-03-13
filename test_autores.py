# from auto_res import funcy, calc_R
import numpy as np
from auto_res_check import calc_R
import matplotlib.pyplot as plt
# import matplotlib
from scipy.stats import linregress
from utils import read_from_excel

plt.style.use("thesis.mplstyle")


def clac_rrr(all_lengths, diameters):
    diameters = np.array(diameters)[::-1] / np.sqrt(
    np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse

    inverse_diameters = 1 / diameters
    all_lengths = np.array(all_lengths)[::-1]
    # return res_int(inverse_diameters, all_lengths)
    return calc_R(all_lengths, inverse_diameters, show=False)

# s_diameters = [0.35,0.42,0.56,0.70,1.0,1.5,2.0,3.0,4.0,6.0,7.9,np.inf] # for spreadsheet
# f_diameters = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]  # NHS data lab folder

# all_diameters = np.array([s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             f_diameters,
#                             f_diameters,
#                             f_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters,
#                             s_diameters])

# all_lengths = np.array([[0.0,2.0,10.8,22.7,39.6,44.9,57.0,65.8,69.3,72.5,75.8,76.8],    # from EPP spreadsheet on teams
#                     [0,2.4,26.5,39.4,52.4,71.3,83.1,85.4,85.5,85,86.2,97],
#                     [0,2.2,27.3,42.1,53.9,72.6,85.6,84.5,86.5,87.5,89.2,96],
#                     [0,2.3,12.2,19.5,29.3,44,57.7,64.5,68.2,69.8,71.3,72.5],
#                     [0,0,12.9,29.7,44.3,62.1,78.9,87.1,113,123,128,139],
#                     [0,0,28.2,44,53.2,73,85.2,91.9,95.5,96.2,96.2,96.2],
#                     [0,0,3.7,19,28.6,52.6,77,93,116.3,145.2,162.5,178.9],
#                     [0,0,6.4,24.4,43.5,58.5,83.9,97.7,132.6,137.9,137.9,137.9],
#                     [0,2.2,12.8,19.1,28.7,39.7,60.9,64.9,65.5,66.6,65.9,65.2],
#                     [0,2.9,26,40.3,55,71,81.3,89.7,93.4,94.6,97.2,98.6],
#                     [0,0,3.8,18.1,23.9,70.1,85.1,128.6,147,165.6,165.1,192],
#                     [0,0,10,31.2,46.4,71.2,85.4,91.8,134,139.1,139.1,139.1],
#                     [0,4.4,14.3,22.1,31.9,41.2,55.9,63.1,67.2,71.7,72.7,73],
#                     [0,0,0,0,0,4.6,47.1,117.2,175.6,195,200.5,215.8],               # from spreadsheet / NHS folder
#                     [0, 0, 0, 0, 0, 22.9, 72.6, 108, 148, 197, 201, 233],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 41.0, 73.6, 121.0, 169.5, 197.1, 203.1, 221.5],
#                     [0.0,0.0,12.9,29.7,44.3,62.1,78.9,87.1,113.0,123.0,128.0,139.0],# from copy of pipe phantom data sheet on teams
#                     [0.0,0.0,0.0,0.0,0.0,22.9,72.6,108.0,148.0,197.0,201.0,233.0],  #
#                     [0.0,1.2,14.9,32.0,36.5,55.4,58.7,58.7,58.7,58.7,58.7,58.7],    #
#                     [0.0,0.0,19.6,31.7,36.8,53.4,53.6,56.0,58.5,58.5,58.5,58.5],    #
#                     [0.0,0.0,0.0,0.0,0.0,40.7,58.3,103.2,157.0,192.5,201.8,221.1],  #
#                     [0.0,0.0,0.0,0.0,0.0,38.1,62.1,123.7,149.3,199.2,204.2,225.8]]) #

# NHS_results = np.array([[70,52,0.75],
#                         [102,68,0.66],
#                         [105,69,0.66],
#                         [65,50,0.77],
#                         [95,81,0.85],
#                         [108,71,0.66],
#                         [87,101,1.15],
#                         [94,91,0.97],
#                         [63,48,0.76],
#                         [105,71,0.67],
#                         [99,120,1.21],
#                         [102,90,0.89],
#                         [68,48,0.71],
#                         [70,179,2.56],
#                         [73.5, 167, 2.27],
#                         [83,163,1.97],
#                         [95,81,0.85],
#                         [74,167,2.27],
#                         [72,48,0.7],
#                         [72,46,0.64],
#                         [77,159,2.1],
#                         [79,164,2.08]])

NHS_results, all_lengths, diameters = read_from_excel()
# print(NHS_results.shape, all_diameters.shape, all_lengths.shape)


EPP_ = []
ST_ = []
percent_difference = []
difference = []
sig_percent_difference = []

num1 = 2
num2 = 0

for lens, results in zip(all_lengths, NHS_results):
    results = results[::-1]
    ST_results = clac_rrr(lens, diameters)
    EPP_.append(results)
    ST_.append(ST_results)
    percent_difference.append(abs(ST_results - results) * 100 / results)
    sig_percent_difference.append((ST_results - results) * 100 / results)
    difference.append(ST_results - results)

difference = np.array(difference)
percent_difference = np.array(percent_difference)
sig_percent_difference = np.array(sig_percent_difference)

# result = linregress(EPP_, ST_)
# print(result)
# print(f"r2 value {(result.rvalue)**2}")

# print(f"average percentage difference: {np.average(percent_difference)}")
# print(f"std of percentage difference: {np.std(percent_difference)}")
# print(f"maximum percentage difference: {np.max(percent_difference)}")

# print(f"average abs difference: {np.average((difference))}")
# print(f"std of difference: {np.std((difference))}")
# print(f"maximum difference: {np.max(abs(difference))}")

# print(len(percent_difference[abs(percent_difference) < 1.7]) * 100 / len(percent_difference))

print(len(percent_difference))

# plt.scatter(EPP_, ST_)
# plt.xlabel("EPP")
# plt.ylabel("ST")
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)

ax1.hist(sig_percent_difference.T[0], bins=80, density=True, color="C0", label="Characteristic resolution")
ax2.hist(sig_percent_difference.T[1], bins=80, density=True, color="C1", label="Depth of field")
ax3.hist(sig_percent_difference.T[2], bins=80, density=True, color="C2", label="Resolution integral")

ax1.set_ylim()
ax1.vlines(0, ax1.get_ylim()[0], ax1.get_ylim()[1], colors="k", linestyles="dashed")
ax2.set_ylim()
ax2.vlines(0, ax2.get_ylim()[0], ax2.get_ylim()[1], colors="k", linestyles="dashed")
ax3.set_ylim()
ax3.vlines(0, ax3.get_ylim()[0], ax3.get_ylim()[1], colors="k", linestyles="dashed")

ax2.set_ylabel("Counts (arb. units)")
ax3.set_xlabel("Difference between code and spreadsheet (%)")

ax1.get_yaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])

ax1.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()
plt.show()