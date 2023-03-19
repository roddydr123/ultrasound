import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# data = pd.read_csv("analysed/gen3/all_data.txt").drop(["Probe","R upper", "R lower", "Dr upper", "Dr lower", "Lr upper", "Lr lower"], axis=1)

# plt.figure(figsize=(10,8))
# sns.set_theme(style="white")
# corr = data.corr()
# heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt='.2g')
# plt.show()

D1 = np.linspace(40,150, 10)
D2 = 1.1 * D1 #+ (np.random.rand(10) * 1)

# D2[0] *= 1.3
# D2[-1] *= 0.87

F1 = np.linspace(1,5,10)
F2 = 0.5 * F1 + (np.random.rand(10) * 3)

# G1 = np.random.rand(10) * 100
# G2 = 4 * G1 + 10

print(pearsonr(D1, D2))
print(pearsonr(D2, F2))
print(pearsonr(F1, F2))
print(pearsonr(D1/F1, D2/F2))

# print(D1, F1)
# print(D1/F1)


plt.scatter(D1, D2)
plt.scatter(F1,F2)
plt.scatter(D1/F1, D2/F2)
plt.show()
