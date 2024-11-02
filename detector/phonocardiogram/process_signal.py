import pickle

from scipy.io import loadmat

mat = loadmat('PCG_dataset.mat')
pcg = mat['PCG_dataset']

x = pcg[0, 0][0]
y = pcg[0, 0][1]
with open("data.pickle", "wb") as f:
    pickle.dump((x, y), f)