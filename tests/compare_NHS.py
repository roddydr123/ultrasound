import pathlib
import sys
p = pathlib.Path(__file__).parents
SCRIPTS_PATH = str(p[1]) + "/sample"
FILES_PATH = str(p[0]) + "/files"
sys.path.insert(0, SCRIPTS_PATH)
import numpy as np
from resolution_integral import calc_R
import matplotlib.pyplot as plt
# import matplotlib
from scipy.stats import linregress
from utils import read_from_excel

# plt.style.use("thesis.mplstyle")


def calc_R_prepper(all_lengths, diameters):
    diameters = np.array(diameters) / np.sqrt(
    np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse

    inverse_diameters = 1 / diameters
    all_lengths = np.array(all_lengths)
    return calc_R(all_lengths, inverse_diameters, show=False)


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
    ST_results = calc_R_prepper(lens, diameters)
    EPP_.append(results)
    ST_.append(ST_results)
    percent_difference.append(abs(ST_results - results) * 100 / results)
    sig_percent_difference.append((ST_results - results) * 100 / results)
    difference.append(ST_results - results)

difference = np.array(difference)
percent_difference = np.array(percent_difference)
sig_percent_difference = np.array(sig_percent_difference)


def plotter():

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

plotter()