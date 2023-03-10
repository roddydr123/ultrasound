import numpy as np



def read_from_excel():
    data = np.genfromtxt("other_data/res_ints_only.csv", dtype=float, delimiter=",")
    NHS_results = data[:, :3]
    all_lengths = data[:, 3:]
    all_diameters = [0.35,0.42,0.56,0.70,1.0,1.5,2.0,3.0,4.0,6.0,7.9,np.inf]
    return NHS_results, all_lengths, all_diameters


if __name__=="__main__":
    read_from_excel()