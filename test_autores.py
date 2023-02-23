# from auto_res import funcy, calc_R
import numpy as np
from auto_res_check import calc_R


def test_autores_check_C14():
    lengths = [0, 0, 0, 0, 0, 22.9, 72.6, 108, 148, 197, 201, 233]  # 4C1 NHS folder
    diameters = np.array(
        [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
    )  # NHS data lab folder

    diameters = np.array(diameters)[::-1] / np.sqrt(
    np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse

    inverse_diameters = 1 / diameters
    lengths = np.array(lengths)[::-1]
    assert calc_R(lengths, inverse_diameters, show=False) == (2.269, 167.5, 73.82)


def test_autores_check_14L5():
    lengths = [0.0, 2.0, 10.8, 22.7, 39.6, 44.9, 57.0, 65.8, 69.3, 72.5, 75.8, 76.8]
    diameters = np.array(
        [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
    )  # NHS data lab folder

    diameters = np.array(diameters)[::-1] / np.sqrt(
    np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse

    inverse_diameters = 1 / diameters
    lengths = np.array(lengths)[::-1]
    assert calc_R(lengths, inverse_diameters, show=False) == (0.75, 52, 70)