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
import pandas as pd
from data_loader import DataGen
from sklearn.utils import shuffle

def get_ids(fold,  EXCEL_PATH, split):
    TRAIN_EXCELS = []
    TEST_EXCELS = []
    DATA_IDS = []
    
    TEST_IDS = []
    for ii in range(1, No_FOLDS + 1):
        EX = "FOLD_{}.xlsx".format(ii)
        if ii == fod:
            TEST_EXCELS.append(EX)
        else:
            TRAIN_EXCELS.append(EX)
    for EXCEL in TRAIN_EXCELS:
        EXCEL_FILE = EXCEL_PATH + "/{}".format(EXCEL)

        data = pd.read_excel(EXCEL_FILE)
        ids = list(data["Name"])
        DATA_IDS = DATA_IDS + ids
    DATA_IDS = shuffle(DATA_IDS, random_state=0)
    EXCEL_FILE = EXCEL_PATH + "/{}".format(TEST_EXCELS[0])
    data = pd.read_excel(EXCEL_FILE)
    ids = list(data["Name"])
    TEST_IDS = TEST_IDS + ids
    train_val_split = int(round(len(DATA_IDS)*split))
    TRAIN_IDS, VAL_IDS = DATA_IDS[:int(round(len(DATA_IDS)*split))], DATA_IDS[int(round(len(DATA_IDS)*split)):]
    
    return TRAIN_IDS, VAL_IDS, TEST_IDS
    
####paths
No_FOLDS = 5

#run_type = "train"
run_type = "inference"

mod = "UNet"
#mod = "unet"
DATASET_PATH = "/scratch/a.cot12/ultrasound_deeplearning/ULTRASOUND_DATASET"
EXCEL_PATH = DATASET_PATH + "/Excel_files"
fold = int(sys.argv[1])
WIEGHTS_SAVE = "C_WIEGHTS"
if os.path.exists(WIEGHTS_SAVE) == False:
    os.mkdir(WIEGHTS_SAVE)
PRED_SAVE = DATASET_PATH + "/Predictions"
if os.path.exists(PRED_SAVE) == False:
    os.mkdir(PRED_SAVE)
PRED_SAVE = PRED_SAVE + "/cory_npy"
if os.path.exists(PRED_SAVE) == False:
    os.mkdir(PRED_SAVE)
IMGS_PATH = DATASET_PATH + "/ALL_IMAGES"

image_size = 224
batch_size = 16
epochs = 50
image_channels = 3
mask_channels = 1
lr = 0.001
SEED = 1
test = "t0"
split = 0.8
size=(image_size, image_size, image_channels)

##SEEDs

os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

train_image_path = os.path.join(IMGS_PATH, "images")
train_mask_path = os.path.join(IMGS_PATH, "masks")


for fod in range(1, No_FOLDS + 1):
    TRAIN_IDS, VAL_IDS, TEST_IDS = get_ids(fod, EXCEL_PATH, split)
    
    if run_type == "train":
        train = DataGen(TRAIN_IDS, train_image_path, train_mask_path, image_size, True)
        val = DataGen(VAL_IDS, train_image_path, train_mask_path, image_size, False)
        X_train, Y_train = train.create_arg_data() ### id, aurgment true/false
        X_val, Y_val = val.create_arg_data()
        print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
    if run_type == "inference":
        test = DataGen(TEST_IDS, train_image_path, train_image_path, image_size, False)
        X_test, Y_test = test.create_arg_data()
        print(X_test.shape, Y_test.shape)
    
    
    
    
    
    WIEGTHS_OUTPUT = os.path.join(WIEGHTS_SAVE, "{}_{}_{}_{}.h5".format(mod, fold, fod,"0"))
    tensorboard = TensorBoard(log_dir= "{}_{}_{}".format(mod, fold, fod))
    model = unet(size = size)
    model.summary()
    checkpointer = ModelCheckpoint(filepath=WIEGTHS_OUTPUT, mode='min', verbose = 1, save_best_only=True, save_weights_only = True)
    METRICS = [dice_coeff, dice_loss, keras.metrics.binary_accuracy, keras.metrics.MeanIoU(num_classes=2)]
    opt = keras.optimizers.Adam(lr)
    model.compile(optimizer = opt, loss=[dice_loss], metrics=[METRICS])
    #model.load_weights(INIT_PATH)
    callbacks = [tensorboard, EarlyStopping(patience = 10, monitor=('val_loss')),ReduceLROnPlateau(monitor=('val_loss'), verbose = 1,factor=0.1, patience=5, cooldown=0, min_lr=0), checkpointer]
    if run_type == "train":
        model.fit(X_train,Y_train, validation_data = (X_val, Y_val), epochs=epochs, callbacks = callbacks, batch_size=batch_size)
    if run_type == "inference":
        model.load_weights(WIEGTHS_OUTPUT)
        #loss_val, dice_val = model.evaluate(X_test, Y_test, verbose=1)
        model_save = PRED_SAVE + "/{}_{}_{}_{}".format(mod, fold, fod,"0")
        if os.path.exists(model_save) == False:
            os.mkdir(model_save)
        result = model.predict(X_test, verbose=1)
        #results_accuracy = (result >= 0.5).astype(np.uint8)
        for ii in range(len(TEST_IDS)):
            prediced_mask = np.squeeze(result[ii])
            save_location_predict_masks = os.path.join(model_save , TEST_IDS[ii] + ".npy")
            #cv2.imwrite(save_location_predict_masks, prediced_mask) 
            np.save(save_location_predict_masks, prediced_mask)
