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



def test_extract_Ls():
    videos = [f"{FILES_PATH}/vid50.txt",
            f"{FILES_PATH}/vid51.txt",
            f"{FILES_PATH}/vid56.txt",
            f"{FILES_PATH}/vid57.txt"]

    inv_diameters = np.linspace(0.01, 1.7, 400)

    pipe_diameters = 1 / inv_diameters

    L_dict, lengths, diameters = extract_Ls(videos, pipe_diameters, 20, 3)

    with open('files/saved_L_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    with open('files/saved_lengths.pkl', 'rb') as f:
        loaded_lengths = pickle.load(f)

    with open('files/saved_diameters.pkl', 'rb') as f:
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
