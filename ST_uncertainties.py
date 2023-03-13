from slice_thickness import extract_Ls
from auto_res_check import calc_R
import numpy as np


videos = [
[27,28,29,30],
[50,51,56,57],
[60,61,62,63], 
[21,22,23,24],
[72,73,74,75], 
[25,26,31,32],
[35,36,37,38],
[68,69,70,71],
[64,65,66,67],
[78,79,80,81]
]

augs = [-0.1, 0, 0.1]

inv_diameters = np.linspace(0.01, 1.7, 400)

pipe_diameters = 1 / inv_diameters[::-1]

all_dat = []

for video in videos:
    for aug in augs:
        L_dict, lengths, diameters = extract_Ls(video, pipe_diameters, 20, 3, aug=aug)

        diameters = np.array(diameters)[::-1]
        inverse_diameters = 1 / diameters

        lengths = np.array(lengths)[::-1]

        data = calc_R(lengths, inverse_diameters, show=False)

        # print(list(map(float, data)))

        all_dat.append([aug] + list(data))

np.savetxt("analysed/gen3/ST_uncertainties.txt", all_dat)