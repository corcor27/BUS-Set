import os
import sys
import random
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



class DataGen(keras.utils.Sequence):
    def __init__(self, dataset_path, dataset, excel_file, excel_path, image_size, augment):
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.excel_path = excel_path
        self.excel_file = excel_file
        self.image_size = image_size
        self.augment = augment
        self.name_path = os.path.join(self.dataset_path, self.dataset)
        self.on_epoch_end()

    def GRADE_CONVERT(self, GRADE):
        
        base = np.zeros((1,2), dtype=int)
        if int(GRADE) == 0:
            base[0,0] = 1
        else:
            base[0,1] = 1
        return base
        
    def __len__(self):
        return int(len(self.ids))
        
    def __load__(self, NAME):
        #Path
        
        IMG_PATH = os.path.join(self.name_path, "images", NAME + ".png")
        MASK_PATH = os.path.join(self.name_path, "masks", NAME + ".png")
        #Read Image
        IMG = cv2.imread(IMG_PATH,0)
        IMG = cv2.resize(IMG, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC) ####set to 224x224
        MASK = cv2.imread(MASK_PATH,0)
        MASK = cv2.resize(MASK, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC) ####set to 224x224
        MASK = MASK.astype(np.bool_)
        NEW_IMAGE = np.zeros((IMG.shape), dtype=np.int8)
        for kk in range(0, MASK.shape[0]):
            for ii in range(0, MASK.shape[1]):
                if MASK[kk,ii] == True:
                    NEW_IMAGE[kk,ii] = IMG[kk,ii]
        #Read Masks
        #Normalizaing
        NEW_IMAGE = NEW_IMAGE/255
        
        return NEW_IMAGE

    def __getitem__(self):
        
        image = []
        GRADE = []
        
        EXCEL_PATH = os.path.join(self.excel_path, self.excel_file)
        data = pd.read_excel(EXCEL_PATH)
        for row in range(0, data.shape[0]):
            CLASS = int(data['Class'].iloc[row])
            GRADE_VALUE = np.squeeze(np.array(self.GRADE_CONVERT(CLASS)))
            _img = self.__load__(str(data['Name'].iloc[row]))
            image.append(_img)
            GRADE.append(GRADE_VALUE)
        if self.augment == True:
            for num in range(0, len(image)):
                image.append(cv2.flip(image[num], 0))
                GRADE.append(GRADE_VALUE[num])
           
        image = np.array(image)
        GRADE  = np.array(GRADE)
        
        
        return image, GRADE
    def create_arg_data(self):
        X_train, Y_train = self.__getitem__()
        
        return X_train, Y_train

    
    def on_epoch_end(self):
        pass

    

    
