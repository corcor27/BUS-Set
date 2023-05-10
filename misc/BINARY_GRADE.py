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
from models_cluster import GRADE_MODEL, ResNet101, SK_ResNet101
import cv2
from data_loader_box import DataGen
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import scipy.ndimage as ndi 
import scipy
import tensorflow as tf
import shutil
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def acc_pr_option(ground, predictions):
    count = []
    for ii in range(0, ground.shape[0]):
        if (predictions[ii,:] == ground[ii,:]).all():
            count.append(int(1))
        else:
            count.append(int(0))

    return (sum(count)/len(count))*100

def confusion_matric(ground, prediction):
    base = np.zeros((4,2))
    for ii in range(0, ground.shape[0]):
        for kk in range(0, ground.shape[1]):
            if ground[ii,kk] == 1:
                if prediction[ii,kk] == 1:
                    base[0,kk] += 1
                else:
                    base[1,kk] += 1
            else:
                if prediction[ii,kk] == 0:
                    base[2,kk] += 1
                else:
                    base[3,kk] += 1
    d = {}
    d["M_TP"] = base[0,0]
    d["M_FP"] = base[1,0]
    d["M_TN"] = base[2,0]
    d["M_FN"] = base[3,0]
    d["B_TP"] = base[0,1]
    d["B_FP"] = base[1,1]
    d["B_TN"] = base[2,1]
    d["B_FN"] = base[3,1]
    return d

WIEGHTS_SAVE = "WIEGHTS"
if os.path.exists(WIEGHTS_SAVE) == False:
    os.mkdir(WIEGHTS_SAVE)

factor = 1.5

FEATURE = "Mag Vs Ben_selective_box"

DATASET_PATH = "ULTRASOUND_DATASET"

EXCEL_PATH = os.path.join(DATASET_PATH, "Excel_files")

EXCEL_FILES = ["DATASET1_TRAIN.xlsx", "DATASET1_VAL.xlsx", "DATASET1_TEST.xlsx"]

####parameters
image_size = 224
batch_size = 10

image_channels = 1
mask_channels = 1
lr = 0.001
SEED = 1
test = "t0"
epochs = 40

##SEEDs

os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

###paths



train = DataGen(DATASET_PATH, "DATASET1_TRAIN", EXCEL_FILES[0], EXCEL_PATH, image_size, factor, True)
val = DataGen(DATASET_PATH, "DATASET1_VAL", EXCEL_FILES[1], EXCEL_PATH, image_size, factor, True)
test = DataGen(DATASET_PATH, "DATASET1_TEST", EXCEL_FILES[2], EXCEL_PATH, image_size, factor, True)
X_train, Y_train = train.create_arg_data() ### id, aurgment true/false
X_val, Y_val = val.create_arg_data()
X_test, Y_test = test.create_arg_data()
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
save_path = "WIEGHTS/{}.h5".format(int(time.time()))
NAME = "DCIS_ATTEMPT-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir= '{}'.format(NAME))
checkpointer = ModelCheckpoint(filepath=save_path, mode='min', verbose = 1, save_best_only=True, save_weights_only = True)
METRICS = [tf.keras.metrics.CategoricalCrossentropy, 'accuarcy', tf.keras.losses.BinaryCrossentropy]
model = ResNet101((image_size,image_size,1))
model.compile(optimizer='adam',loss="binary_crossentropy", metrics=['accuracy'])
callbacks = [tensorboard, checkpointer, ReduceLROnPlateau(monitor=('val_loss'), factor=0.7, patience=10, cooldown=0, min_lr=0)]
model.summary()
model.fit(X_train,Y_train, validation_data = (X_val, Y_val), epochs=epochs, callbacks = callbacks, batch_size=batch_size)


result = model.predict(X_test, verbose=1)
results_accuracy = (result >= 0.5).astype(np.uint8)

score = acc_pr_option(Y_test, results_accuracy)
TP_FP_VALS = confusion_matric(Y_test, results_accuracy)

WIEGHT_NAME = save_path.replace("WIEGHTS/", " ")

OUT_EXC = "RECORD/MAGVBEN_RECORD_augment.xlsx"


if os.path.exists(OUT_EXC) == False:
    d = {'feature': FEATURE, 'wieghts name': WIEGHT_NAME, 'size': image_size, 'scale_factor':factor ,'batch': batch_size, 'lr': lr, 'epochs': epochs, "ACC" : score}
    d.update(TP_FP_VALS)
    data = pd.DataFrame([d])
    data.to_excel(OUT_EXC)
    

else:
    data = pd.read_excel(OUT_EXC, index_col=[0])
    d = {'feature': FEATURE, 'wieghts name': WIEGHT_NAME, 'size': image_size, 'scale_factor':factor , 'batch': batch_size, 'lr': lr, 'epochs': epochs, "ACC" : score}
    d.update(TP_FP_VALS)
    new_data = data.append([d], ignore_index = True)
    new_data.to_excel(OUT_EXC)



