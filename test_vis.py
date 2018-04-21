import h5py

with h5py.File('reconstruction.hdf5', 'r') as f:
    recons_data = f['recon_mrc'][...]

print(recons_data.shape)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

output_path = "./output"
num_img = 10
for i in range(num_img):
    voxels = recons_data[i]
    x, y, z = np.indices((32, 32, 32))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
    filename = 'demo' + str(i) + '.png'
    path = os.path.join(output_path, filename)
    plt.savefig(path)