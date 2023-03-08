# from auto_res import funcy, calc_R
import numpy as np
from auto_res_check import calc_R
import matplotlib.pyplot as plt
from scipy.stats import linregress


# def test_autores_check_C14():
#     lengths = [0, 0, 0, 0, 0, 22.9, 72.6, 108, 148, 197, 201, 233]  # 4C1 NHS folder
#     diameters = np.array(
#         [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
#     )  # NHS data lab folder

#     diameters = np.array(diameters)[::-1] / np.sqrt(
#     np.cos(np.deg2rad(40))
#     )  # convert to effective diameter and reverse

#     inverse_diameters = 1 / diameters
#     lengths = np.array(lengths)[::-1]
#     assert calc_R(lengths, inverse_diameters, show=False) == (2.269, 167.5, 73.82)


# def test_autores_check_14L5():
#     lengths = [0.0, 2.0, 10.8, 22.7, 39.6, 44.9, 57.0, 65.8, 69.3, 72.5, 75.8, 76.8]
#     diameters = np.array(
#         [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
#     )  # NHS data lab folder

#     diameters = np.array(diameters)[::-1] / np.sqrt(
#     np.cos(np.deg2rad(40))
#     )  # convert to effective diameter and reverse

#     inverse_diameters = 1 / diameters
#     lengths = np.array(lengths)[::-1]
#     assert calc_R(lengths, inverse_diameters, show=False) == (0.75, 52, 70)

def clac_rrr(lengths, diameters):
    diameters = np.array(diameters)[::-1] / np.sqrt(
    np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse

    inverse_diameters = 1 / diameters
    lengths = np.array(lengths)[::-1]
    return calc_R(lengths, inverse_diameters, show=False)

s_diameters = [0.35,0.42,0.56,0.70,1.0,1.5,2.0,3.0,4.0,6.0,7.9,np.inf] # for spreadsheet
f_diameters = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]  # NHS data lab folder

all_diameters = np.array([s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            s_diameters,
                            f_diameters,
                            f_diameters,
                            f_diameters])

lengths = np.array([[0.0,2.0,10.8,22.7,39.6,44.9,57.0,65.8,69.3,72.5,75.8,76.8],
                    [0,2.4,26.5,39.4,52.4,71.3,83.1,85.4,85.5,85,86.2,97],
                    [0,2.2,27.3,42.1,53.9,72.6,85.6,84.5,86.5,87.5,89.2,96],
                    [0,2.3,12.2,19.5,29.3,44,57.7,64.5,68.2,69.8,71.3,72.5],
                    [0,0,12.9,29.7,44.3,62.1,78.9,87.1,113,123,128,139],
                    [0,0,28.2,44,53.2,73,85.2,91.9,95.5,96.2,96.2,96.2],
                    [0,0,3.7,19,28.6,52.6,77,93,116.3,145.2,162.5,178.9],
                    [0,0,6.4,24.4,43.5,58.5,83.9,97.7,132.6,137.9,137.9,137.9],
                    [0,2.2,12.8,19.1,28.7,39.7,60.9,64.9,65.5,66.6,65.9,65.2],
                    [0,2.9,26,40.3,55,71,81.3,89.7,93.4,94.6,97.2,98.6],
                    [0,0,3.8,18.1,23.9,70.1,85.1,128.6,147,165.6,165.1,192],
                    [0,0,10,31.2,46.4,71.2,85.4,91.8,134,139.1,139.1,139.1],
                    [0,4.4,14.3,22.1,31.9,41.2,55.9,63.1,67.2,71.7,72.7,73],
                    [0,0,0,0,0,4.6,47.1,117.2,175.6,195,200.5,215.8],
                    [0, 0, 0, 0, 0, 22.9, 72.6, 108, 148, 197, 201, 233],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 41.0, 73.6, 121.0, 169.5, 197.1, 203.1, 221.5]])

NHS_results = np.array([[70,52,0.75],
                        [102,68,0.66],
                        [105,69,0.66],
                        [65,50,0.77],
                        [95,81,0.85],
                        [108,71,0.66],
                        [87,101,1.15],
                        [94,91,0.97],
                        [63,48,0.76],
                        [105,71,0.67],
                        [99,120,1.21],
                        [102,90,0.89],
                        [68,48,0.71],
                        [70,179,2.56],
                        [73.5, 167, 2.27],
                        [88,163,1.97]])
    

EPP_ = []
ST_ = []
percent_difference = []

for diameters, lens, results in zip(all_diameters[:-1], lengths[:-1], NHS_results[:-1]):
    ST_results = clac_rrr(lens, diameters)
    EPP_.append(results[2])
    ST_.append(ST_results[0])
    percent_difference.append(abs(results[2] - ST_results[0]) * 100 / results[2])

result = linregress(EPP_, ST_)
print(result)
print(f"r2 value {(result.rvalue)**2}")

print(f"average percentage difference: {np.average(percent_difference)}")
print(f"std of percentage difference: {np.std(percent_difference)}")
print(f"maximum percentage difference: {np.max(percent_difference)}")

plt.scatter(EPP_, ST_)
plt.show()