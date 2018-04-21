import numpy as np
import h5py
import glob
import os
import scipy.io
from random import shuffle
import random

SEED = 448
random.seed(SEED)

input_path = '/Users/jingjingy/Python/PycharmProject/ShapeNets/data'
label = {
    'airplane': 0,
    'bathtub': 1,
    'bed': 2,
    'bottle': 3,
    'chair': 4,
    'cup': 5,
    'guitar': 6,
    'laptop': 7,
    'stairs': 8,
    'toilet': 9
}
'''
label = {
    'chair': 1
}
'''
train_addrs = []
labels = []
combine_train = []
combine_test = []
for item in os.listdir(input_path):
    if item != '.DS_Store':
        current_label = label.get(item)
        current_dir = os.path.join(input_path, item, '30')
        train_path = os.path.join(current_dir, 'train')
        test_path = os.path.join(current_dir, 'test')
        train_addrs = glob.glob(train_path + '/*.mat')
        test_addrs = glob.glob(test_path + '/*.mat')
        train_labels = np.full(len(train_addrs), current_label, dtype=int)
        print('Training data for %s: %d' % (item, len(train_addrs)))
        test_labels = np.full(len(test_addrs), current_label, dtype=int)
        print('Testing data for %s: %d' % (item, len(test_addrs)))
        temp_train_addrs = list(zip(train_addrs, train_labels))
        temp_test_addrs = list(zip(test_addrs, test_labels))
        combine_train += temp_train_addrs
        combine_test += temp_test_addrs


shuffle(combine_train)
shuffle(combine_test)

mat_train, train_label = zip(*combine_train)
mat_test, test_label = zip(*combine_test)

from sklearn.model_selection import train_test_split
mat_test, mat_val, test_label, val_label = train_test_split(mat_test, test_label, test_size=0.5)

print('Total training example: %d' % (len(mat_train)))
print('Total testing example: %d' % (len(mat_test)))
print('Total validation example: %d' % (len(mat_val)))

train_file = open('train.txt', 'w')
test_file = open('test.txt', 'w')
val_file = open('validation.txt', 'w')

box_size = 32
train_shape = (len(mat_train), box_size, box_size, box_size)
test_shape = (len(mat_test), box_size, box_size, box_size)
val_shape = (len(mat_val), box_size, box_size, box_size)

hdf5_file = h5py.File("object.hdf5", "w")
hdf5_file.create_dataset("train_mat", train_shape, np.int8)
hdf5_file.create_dataset("test_mat", test_shape, np.int8)
hdf5_file.create_dataset("val_mat", val_shape, np.int8)

hdf5_file.create_dataset("train_label", (len(train_label), 1), np.int8)
hdf5_file.create_dataset("test_label",  (len(test_label), 1), np.int8)
hdf5_file.create_dataset("val_label",   (len(val_label), 1), np.int8)

voxels = np.zeros((32, 32, 32))
for i in range(len(mat_train)):
    if i % 50 == 0:
        print('Training writing has finished: %d/%d' % (i, len(mat_train)))
    mat = scipy.io.loadmat(mat_train[i])
    voxels[1:31, 1:31, 1:31] = mat['instance']
    hdf5_file["train_mat"][i, ...] = voxels
    hdf5_file["train_label"][i] = train_label[i]
    train_file.write("%s, %d\n" % (mat_train[i], train_label[i]))
print('Training writing has finished...')

for j in range(len(mat_test)):
    if j % 50 == 0:
        print('Testing writing has finished: %d/%d' % (j, len(mat_test)))
    mat = scipy.io.loadmat(mat_test[j])
    voxels[1:31, 1:31, 1:31] = mat['instance']
    hdf5_file["test_mat"][j, ...] = voxels
    hdf5_file["test_label"][j] = test_label[j]
    test_file.write("%s, %d\n" % (mat_test[j], test_label[j]))
print('Testing writing has finished...')

for k in range(len(mat_val)):
    if k % 50 == 0:
        print('Validation writing has finished: %d/%d' % (k, len(mat_val)))
    mat = scipy.io.loadmat(mat_val[k])
    voxels[1:31, 1:31, 1:31] = mat['instance']
    hdf5_file["val_mat"][k, ...] = voxels
    hdf5_file["val_label"][k] = val_label[k]
    val_file.write("%s, %d\n" % (mat_val[k], val_label[k]))
print('Validation writing has finished...')





