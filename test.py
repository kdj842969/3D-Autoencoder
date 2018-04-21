from keras.models import load_model
import h5py
import numpy as np
import mrcfile as mrc
import os

from keras import backend as K
K.clear_session()

with h5py.File('object.hdf5', 'r') as f:
    train_data = f['train_mat'][...]
    val_data = f['val_mat'][...]
    test_data = f['test_mat'][...]

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

train_num = train_data.shape[0]
val_num = val_data.shape[0]
test_num = test_data.shape[0]
box_size = train_data.shape[1]

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])
val_data = val_data.reshape([-1, box_size, box_size, box_size, 1])
test_data = test_data.reshape([-1, box_size, box_size, box_size, 1])

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

autoencoder = load_model('autoencoder.h5')
decoded_imgs = autoencoder.predict(test_data, batch_size=100)
decoded_imgs = decoded_imgs.reshape(test_num, box_size, box_size, box_size)
print("decoded imgs shape is:")
print(decoded_imgs.shape)

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# write back to hdf5 file
hdf5_file = h5py.File("reconstruction.hdf5", "w")
hdf5_file.create_dataset("recon_mat", decoded_imgs.shape, np.int8)
for i in range(len(decoded_imgs)):
    hdf5_file["recon_mat"][i] = decoded_imgs[i]

hdf5_file.close()
print('Reconstruction HDF5 file successfully created.')