import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import add, multiply, subtract
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, UpSampling2D, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import os
import sys
import random
import matplotlib.pyplot as plt
import time
from models import dice_coeff, dice_loss, dice_coef_np, voe_coef_np, selective_unet, unet, denseunet, Unetplusplus
import cv2
from data_loader import DataGen

image_size = 224
batch_size = 16
epochs = 50
image_channels = 1
mask_channels = 1
lr = 0.001
SEED = 1
test = "t0"

##SEEDs

os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

###paths
NAME = "MAGSKUNet{}".format(int(time.time()))

X_train, Y_train = np.load("X_TRAIN.npy"),np.load("Y_TRAIN.npy").astype(bool)
X_test, Y_test = np.load("X_VAL.npy"),np.load("Y_VAL.npy").astype(bool)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

tensorboard = TensorBoard(log_dir= '{}'.format(NAME))


save_path = "WIEGHTS/{}.h5".format(int(time.time()))
NAME = "DCIS_ATTEMPT-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir= '{}'.format(NAME))
checkpointer = ModelCheckpoint(filepath=save_path, monitor='val_loss',mode='min', verbose = 1, save_best_only=True, save_weights_only = True)
METRICS = [dice_coeff, dice_loss, keras.metrics.binary_accuracy, keras.metrics.MeanIoU(num_classes=2)]
optimizer = keras.optimizers.Adam(learning_rate=lr)
model = selective_unet()
model.compile(optimizer=optimizer,loss=[dice_loss], metrics=[METRICS])
callbacks = [tensorboard, checkpointer, ReduceLROnPlateau(monitor=('val_loss'), factor=0.5, patience=5, cooldown=0, min_lr=0)]
model.summary()
model.fit(X_train,Y_train, validation_data = (X_test, Y_test), epochs=epochs, callbacks = callbacks, batch_size=batch_size)


