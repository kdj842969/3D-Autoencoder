import h5py
from time import time
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


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

input_img = Input(shape=(32, 32, 32, 1))
x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
# x = Convolution3D(120, (5, 5, 5), activation='relu', padding='same')(x)
# encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)

print("shape of encoded: ")
print(K.int_shape(encoded))

# x = Convolution3D(16, (5, 5, 5), activation='relu', padding='same')(encoded)
# x = UpSampling3D((2, 2, 2))(x)
x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(encoded)
x = UpSampling3D((2, 2, 2))(x)
x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
decoded = Convolution3D(1, (5, 5, 5), activation='relu', padding='same')(x)
print("shape of decoded: ")
print(K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

autoencoder.fit(train_data, train_data,
              epochs=200,
              batch_size=100,
              validation_data=(val_data, val_data),
              callbacks=[tensorboard])

autoencoder.save('autoencoder.h5')
print("Training finished...")