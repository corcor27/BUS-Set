import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import add, multiply, subtract
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, \
    UpSampling2D, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, concatenate, \
    BatchNormalization, Concatenate
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import random
import matplotlib.pyplot as plt
import time
from new_models_cluster import GRADE_MODEL, ResNet101
import cv2
from new_data_loader_cluster import DataGen
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import scipy.ndimage as ndi 
import scipy
import tensorflow as tf
import shutil
from sklearn.manifold import TSNE
import sys

WIEGHTS_SAVE = "WIEGHTS"
if os.path.exists(WIEGHTS_SAVE) == False:
    os.mkdir(WIEGHTS_SAVE)

FEATURE = "Mag Vs Ben_diff"

DATASET_PATH = "ULTRASOUND_DATASET"

EXCEL_PATH = os.path.join(DATASET_PATH, "Excel_files")

EXCEL_FILES = ["DATASET1_TRAIN.xlsx", "DATASET1_VAL.xlsx", "DATASET1_TEST.xlsx"]

####parameters
image_size = 224
batch_size = 6

image_channels = 1
mask_channels = 1
lr = 0.001
SEED = 1
test = "t0"
epochs = 30

##SEEDs

os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

###paths



train = DataGen(DATASET_PATH, "DATASET1_TRAIN", EXCEL_FILES[0], EXCEL_PATH, image_size, False)
val = DataGen(DATASET_PATH, "DATASET1_VAL", EXCEL_FILES[1], EXCEL_PATH, image_size, False)
test = DataGen(DATASET_PATH, "DATASET1_TEST", EXCEL_FILES[2], EXCEL_PATH, image_size, False)
X_train, Y_train = train.create_arg_data() ### id, aurgment true/false
X_val, Y_val = val.create_arg_data()
X_test, Y_test = test.create_arg_data()
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
save_path = "WIEGHTS/{}.h5".format(int(time.time()))
NAME = "DCIS_ATTEMPT-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir= '{}'.format(NAME))
checkpointer = ModelCheckpoint(filepath=save_path, mode='min', verbose = 1, save_best_only=True, save_weights_only = True)
METRICS = [tf.keras.metrics.CategoricalCrossentropy, 'accuarcy']
model = ResNet101((image_size,image_size,1))
model.compile(optimizer='adam',loss="binary_crossentropy", metrics=['accuracy'])
callbacks = [tensorboard, checkpointer]
model.fit(X_train,Y_train, validation_data = (X_val, Y_val), epochs=epochs, callbacks = callbacks, batch_size=batch_size)


def prepare_data(features):
    base = np.zeros((features.shape[0], features.shape[-1]))
    for fet in range(0, features.shape[0]):
        a_f = features[fet,:,:,:].flatten()
        base[fet, :] = a_f
    return base
    
test_features = model.predict(X_test)
train_features = model.predict(X_train)
val_features = model.predict(X_val)

all_features = np.concatenate((train_features,val_features, test_features), axis=0)

labels =  np.concatenate((Y_train, Y_val, Y_test), axis=0)
print(all_features.shape)
print(labels.shape)

extracted_features = prepare_data(all_features)
print(extracted_features.shape)

out = "cluster_" + TYPE_ARRAY[TYPE] + ".png"
cluster = TSNE(n_components=2).fit_transform(extracted_features)
print(cluster.shape)
print(labels.shape)
mag = []
ben = []
for ii in range(0,labels.shape[0]):
    if labels[ii] == 1:
        mag.append(cluster[ii,:])
    else:
        ben.append(cluster[ii,:])

new_mag = np.array(mag)
new_ben = np.array(ben)
print(new_mag.shape, new_ben.shape)
plt.subplot(2, 2, 1)
plt.scatter(new_mag[:,0], new_mag[:,1], label= TYPE_ARRAY[TYPE])
plt.legend()
plt.subplot(2, 2, 2)
plt.scatter(new_ben[:,0], new_ben[:,1], label='OT' )
plt.legend()
plt.subplot(2, 2, 3)
plt.scatter(new_mag[:,0], new_mag[:,1], label=TYPE_ARRAY[TYPE] )
plt.scatter(new_ben[:,0], new_ben[:,1], label='OT' )
plt.legend()
plt.savefig(out)
