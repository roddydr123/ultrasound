import pytest
import numpy as np
import pickle
import sys
import pathlib

p = pathlib.Path(__file__).parents
SCRIPTS_PATH = str(p[1]) + "/sample"
FILES_PATH = str(p[0]) + "/files"
sys.path.insert(0, SCRIPTS_PATH)

from slice_thickness import extract_Ls
from resolution_integral import calc_R
from videos import Video



def test_extract_Ls():
    videos = [f"{FILES_PATH}/vid50.txt",
            f"{FILES_PATH}/vid51.txt",
            f"{FILES_PATH}/vid56.txt",
            f"{FILES_PATH}/vid57.txt"]

    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters

    L_dict, lengths, diameters = extract_Ls(videos, pipe_diameters, 20, 3)

    with open(f'{FILES_PATH}/saved_L_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    with open(f'{FILES_PATH}/saved_lengths.pkl', 'rb') as f:
        loaded_lengths = pickle.load(f)

    with open(f'{FILES_PATH}/saved_diameters.pkl', 'rb') as f:
        loaded_diameters = pickle.load(f)

    assert L_dict == loaded_dict
    assert lengths == loaded_lengths
    assert np.array_equal(diameters, loaded_diameters)


@pytest.mark.parametrize("video_numbs,Resints", [([50,51,56,57],(2.679, 72.23, 26.96)), ([64,65,66,67], (5.477, 144.23, 26.34)), ([21,22,23,24], (2.221, 86.8, 39.08))])
def test_calc_R_and_extract_Ls(video_numbs, Resints):
    videos = [f"{FILES_PATH}/vid{video_numbs[0]}.txt",
            f"{FILES_PATH}/vid{video_numbs[1]}.txt",
            f"{FILES_PATH}/vid{video_numbs[2]}.txt",
            f"{FILES_PATH}/vid{video_numbs[3]}.txt"]

    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters

    L_dict, lengths, diameters = extract_Ls(videos, pipe_diameters, 20, 3)

    inverse_diameters = 1 / diameters

    lengths = np.array(lengths)

    assert calc_R(lengths, inverse_diameters, show=False) == Resints


def test_slice_thickness():
    """Test whether the heights, widths, and depths of the slice thickness peaks extracted
    from the vid50.mp4 video are what they should be."""

    viddata = {"filepath": FILES_PATH, "filenumber": "50", "filename": "vid50.mp4"}
    vid = Video(viddata)

    widths, depths, heights = vid.get_slice_thickness_data(100)

    data = np.array([depths, widths, heights]).T

    test_data = np.load(f"{FILES_PATH}/vid50_test.npy")

    assert np.array_equal(data, test_data)


def calc_R_prepper(all_lengths, diameters):
    diameters = np.array(diameters) / np.sqrt(
    np.cos(np.deg2rad(40))
    )  # convert to effective diameter and reverse

    inverse_diameters = 1 / diameters
    all_lengths = np.array(all_lengths)
    return calc_R(all_lengths, inverse_diameters, show=False)


def test_calc_R_EPP_data():
    """Use the length and diameter data from the NHS spreadsheets. Test to make
    sure the way they're calculated by calc_R has not changed."""

    NHS_results, all_lengths, diameters = read_from_excel(FILES_PATH + "/res_ints_only.csv")
    ST_ = []

    for lens in all_lengths:
        ST_results = calc_R_prepper(lens, diameters)
        ST_.append(ST_results)

    test_data = np.load(f"{FILES_PATH}/EPP_calc_R_results.npy")

    assert np.array_equal(ST_, test_data)


def read_from_excel(filepath):
    data = np.genfromtxt(filepath, dtype=float, delimiter=",")
    NHS_results = data[:, :3]
    all_lengths = data[:, 3:]
    all_diameters = [0.35,0.42,0.56,0.70,1.0,1.5,2.0,3.0,4.0,6.0,7.9,np.inf]
    return NHS_results, all_lengths, all_diameters


def test_calc_R_simple():
    lengths = [0.0,0.0,0.0,6.1,13.9,17.7,23.2,27.8,27.8,27.8,27.8,28.0]

    inverse_diameters = np.linspace(0, 1.7, 12).tolist()

    assert calc_R(lengths, inverse_diameters, show=False) == (0.906, 25.95, 28.66)