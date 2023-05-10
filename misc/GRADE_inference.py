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
from models_cluster import GRADE_MODEL, ResNet101
import cv2
from data_loader_cluster import DataGen
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
    return base


WIEGHTS_SAVE = "WIEGHTS"
if os.path.exists(WIEGHTS_SAVE) == False:
    os.mkdir(WIEGHTS_SAVE)


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
epochs = 20

##SEEDs

os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

###paths




test = DataGen(DATASET_PATH, "DATASET1_TEST", EXCEL_FILES[2], EXCEL_PATH, image_size, False)
X_test, Y_test = test.create_arg_data()
print(X_test.shape, Y_test.shape)

M_TP = []
M_FP = []
M_TN = []
M_FN = []
B_TP = []
B_FP = []
B_TN = []
B_FN = []
EXCEL_PATH = "RECORD/MAGVBEN_RECORD.xlsx"
data = pd.read_excel(EXCEL_PATH)
for row in range(0, data.shape[0]):
    WIEGHT_NAME = str(data['wieghts name'].iloc[row]).strip()
    
    WIEGHTS_LOAD_PATH = os.path.join(WIEGHTS_SAVE, WIEGHT_NAME)
    save_path = "WIEGHTS/{}.h5".format(int(time.time()))
    NAME = "DCIS_ATTEMPT-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir= '{}'.format(NAME))
    checkpointer = ModelCheckpoint(filepath=save_path, mode='min', verbose = 1, save_best_only=True, save_weights_only = True)
    METRICS = [tf.keras.metrics.CategoricalCrossentropy, 'accuarcy', tf.keras.losses.BinaryCrossentropy]
    model = ResNet101((image_size,image_size,1))
    model.compile(optimizer='adam',loss="binary_crossentropy", metrics=['accuracy'])
    callbacks = [tensorboard, checkpointer]
    #model.fit(X_train,Y_train, validation_data = (X_val, Y_val), epochs=epochs, callbacks = callbacks, batch_size=batch_size)
    model.load_weights(WIEGHTS_LOAD_PATH)

    result = model.predict(X_test, verbose=1)
    results_accuracy = (result >= 0.5).astype(np.uint8)

    score = acc_pr_option(Y_test, results_accuracy)
    con_mat = confusion_matric(Y_test, results_accuracy)
    print(WIEGHT_NAME, score)
    M_TP.append(con_mat[0,0])
    M_FP.append(con_mat[1,0])
    M_TN.append(con_mat[2,0])
    M_FN.append(con_mat[3,0])
    B_TP.append(con_mat[0,1])
    B_FP.append(con_mat[1,1])
    B_TN.append(con_mat[2,1])
    B_FN.append(con_mat[3,1])
    print(WIEGHT_NAME,score)

data["M_TP"] = M_TP
data["M_FP"] = M_FP
data["M_TN"] = M_TN
data["M_FN"] = M_FN
data["B_TP"] = B_TP
data["B_FP"] = B_FP
data["B_TN"] = B_TN
data["B_FN"] = B_FN

data.to_excel(EXCEL_PATH)
