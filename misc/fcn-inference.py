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
from models import dice_coeff, dice_loss, dice_coef_np, voe_coef_np, selective_unet, unet, denseunet, Unetplusplus, fcn_dilated
import cv2
from data_loader import DataGen


####paths

TRAIN_TEST_WIEGHT_SAVE = ["DATASET1_TRAIN","DATASET1_VAL","WIEGHTS/ATTUNET"]
#TRAIN_TEST_WIEGHT_SAVE = ["BUSI_TRAIN","BUSI_TEST","WIEGHTS/SK-UNET/BUSI"]
DATASET_PATH = "/home/a.cot12/ultrasound_deeplearning/ULTRASOUND_DATASET"
TRAIN_PATH = os.path.join(DATASET_PATH, TRAIN_TEST_WIEGHT_SAVE[0])
TEST_PATH = os.path.join(DATASET_PATH, TRAIN_TEST_WIEGHT_SAVE[1])
WIEGTHS_OUT_PATH = os.path.join(DATASET_PATH, TRAIN_TEST_WIEGHT_SAVE[2])
NAME = "FCN{}".format(int(time.time()))
WIEGTHS_OUTPUT = os.path.join(WIEGTHS_OUT_PATH, NAME + ".h5")
#INIT_PATH = "/home/a.cot12/ultrasound_deeplearning/init_files/ATTUNet-best.h5"

####parameters
image_size = 224
batch_size = 16
epochs = 100
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
train_image_path = os.path.join(TRAIN_PATH, "images")
train_mask_path = os.path.join(TRAIN_PATH, "masks")
test_image_path = os.path.join(TEST_PATH, "images")
test_mask_path = os.path.join(TEST_PATH, "masks")
train_ids = os.listdir(train_image_path)
test_ids = os.listdir(test_image_path)


"""
for Datagen requires 6 inputs ids, path to images, path to masks, image size, augmentation: True/False
"""

train = DataGen(train_ids, train_image_path, train_mask_path, image_size, True)
test = DataGen(test_ids, test_image_path, test_mask_path, image_size, False)

X_train, Y_train = train.create_arg_data() ### id, aurgment true/false
X_test, Y_test = test.create_arg_data()
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)




tensorboard = TensorBoard(log_dir= '{}'.format(NAME))
model = fcn_dilated()
model.summary()
checkpointer = ModelCheckpoint(filepath=WIEGTHS_OUTPUT, mode='min', verbose = 1, save_best_only=True, save_weights_only = True)
METRICS = [dice_coeff, dice_loss, keras.metrics.binary_accuracy, keras.metrics.MeanIoU(num_classes=2)]
opt = keras.optimizers.Adam(lr)
model.compile(optimizer = opt, loss=[dice_loss], metrics=[METRICS])
#model.load_weights(INIT_PATH)
callbacks = [tensorboard, EarlyStopping(patience = 15, monitor=('loss')),ReduceLROnPlateau(monitor=('loss'), factor=0.1, patience=10, cooldown=0, min_lr=0), checkpointer]
model.fit(X_train,Y_train, validation_data = (X_test, Y_test), epochs=epochs, callbacks = callbacks, batch_size=batch_size)



