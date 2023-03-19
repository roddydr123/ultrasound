from slice_thickness import extract_Ls
from auto_res_check import calc_R
import numpy as np


names = ["9L-D",
        "ML6-15-D",
        "14L5",
        "9L4 (a)",
        "9L4 (b)",
        "C1-5-D",
        "18L6 HD (a)",
        "18L6 HD (b)",
        "4C1",
        "6C1"]


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

EPP_data = np.loadtxt("analysed/gen3/EPP_data.txt", dtype=str, skiprows=1)[:, 1:].astype(float)

augs = [0, -0.1, 0.1]

inv_diameters = np.linspace(0.01, 1.7, 400)

pipe_diameters = 1 / inv_diameters[::-1]

all_dat = []

for probe, name, probe_EPP in zip(videos, names, EPP_data):
    probe_Rs = []
    probe_Drs = []
    probe_Lrs = []
    for aug in augs:
        L_dict, lengths, diameters = extract_Ls(probe, pipe_diameters, 20, 3, aug=aug)

        diameters = np.array(diameters)[::-1]
        inverse_diameters = 1 / diameters

        lengths = np.array(lengths)[::-1]

        Dr, Lr, R = calc_R(lengths, inverse_diameters, show=False)
        bounds = 0.1
        assert (R > round(Lr / Dr, 2) - bounds and R < round(Lr / Dr, 2) + bounds)
        probe_Rs.append(R)
        probe_Drs.append(Dr)
        probe_Lrs.append(Lr)

    # save in format: probe name, EPP R, EPP Dr, EPP Lr,
    # R, R upper, R lower,
    # Dr, Dr upper, Dr lower,
    # Lr, Lr upper, Lr lower
    probe_dat = [name, probe_EPP[0], probe_EPP[2], probe_EPP[1],
                 np.median(probe_Rs), max(probe_Rs) - np.median(probe_Rs), np.median(probe_Rs) - min(probe_Rs),
                 np.median(probe_Drs), max(probe_Drs) - np.median(probe_Drs), np.median(probe_Drs) - min(probe_Drs),
                 np.median(probe_Lrs), max(probe_Lrs) - np.median(probe_Lrs), np.median(probe_Lrs) - min(probe_Lrs)]

    # round the numbers to 5 decimal places
    probe_dat = [name] + list(map(round, probe_dat[1:], [5]*len(probe_dat[1:])))
    all_dat.append(probe_dat)

with open("analysed/gen3/all_data.txt", "w") as f:
    f.write("Probe,EPP R,EPP Dr,EPP Lr,R,R upper,R lower,Dr,Dr upper,Dr lower,Lr,Lr upper,Lr lower\n")
    for line in all_dat:
        strline = [str(x) for x in line]
        f.write(",".join(strline) + "\n")