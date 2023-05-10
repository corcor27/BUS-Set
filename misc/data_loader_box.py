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
    def __init__(self, dataset_path, dataset, excel_file, excel_path, image_size, factor, augment, COLOUR):
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.excel_path = excel_path
        self.excel_file = excel_file
        self.image_size = image_size
        self.augment = augment
        self.name_path = os.path.join(self.dataset_path, self.dataset)
        self.factor = factor
        self.COLOUR = COLOUR
        self.on_epoch_end()

    def GRADE_CONVERT(self, GRADE):
        
        base = np.zeros((1,2), dtype=int)
        if int(GRADE) == 1:
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
        #IMG = cv2.resize(IMG, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC) ####set to 224x224
        MASK = cv2.imread(MASK_PATH,0)
        #MASK = cv2.resize(MASK, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC) ####set to 224x224
        MASK = MASK.astype(np.bool_)
        ROI_CREATE = self.BOUNDING_BOX(IMG, MASK)
        #Normalizaing
        ROI_CREATE = ROI_CREATE
        
        return ROI_CREATE
    
    def dimensions_check(self, Z_START, X_START):
        base = np.zeros((1,2))
        if Z_START >= 0:
            base[0,0] = 0
        else:
            base[0,0] = int(Z_START)
        if X_START >= 0:
            base[0,1] = 0
        else:
            base[0,1] = int(X_START)
        return base
    
        


    def BOUNDING_BOX(self,IMG, MASK):
        X_POS = []
        Z_POS = []
        for kk in range(0, MASK.shape[0]):
            for ii in range(0, MASK.shape[1]):
                if MASK[kk,ii] == True:
                    X_POS.append(ii)
                    Z_POS.append(kk)
                
        MAX_X = max(X_POS)
        MAX_Z = max(Z_POS)
        MIN_X = min(X_POS)
        MIN_Z = min(Z_POS)
        
        DIFF_X = MAX_X - MIN_X
        DIFF_Z = MAX_Z - MIN_Z
        
        X_DIFF_SCALED = int(round(DIFF_X*self.factor, 0))
        Z_DIFF_SCALED = int(round(DIFF_Z*self.factor, 0))
        
            
        
        X_START = int(round(MIN_X - ((X_DIFF_SCALED - DIFF_X)/2), 0))
        Z_START = int(round(MIN_Z - ((Z_DIFF_SCALED - DIFF_Z)/2), 0))
        if X_START < 0 or Z_START < 0:
            IMG_OFFSET = self.dimensions_check(Z_START, X_START)
            
            NEW_X = int(X_START - IMG_OFFSET[0,1])
            NEW_Z = int(Z_START - IMG_OFFSET[0,0])
            
            BOX_IMAGE  = IMG[NEW_Z:NEW_Z + Z_DIFF_SCALED, NEW_X:NEW_X + X_DIFF_SCALED]
            BOX_IMAGE = cv2.resize(BOX_IMAGE, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)
            return BOX_IMAGE
            
        else:
            BOX_IMAGE  = IMG[Z_START:Z_START + Z_DIFF_SCALED, X_START:X_START + X_DIFF_SCALED]
            BOX_IMAGE = cv2.resize(BOX_IMAGE, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)
            return BOX_IMAGE

    def __getitem__(self):
        
        image = []
        GRADE = []
        
        EXCEL_PATH = os.path.join(self.excel_path, self.excel_file)
        data = pd.read_excel(EXCEL_PATH)
        for row in range(0, data.shape[0]):
            CLASS = int(data['Class'].iloc[row])
            GRADE_VALUE = np.squeeze(np.array(self.GRADE_CONVERT(CLASS)))
            _img = self.__load__(str(data['Name'].iloc[row]))
            GRADE.append(GRADE_VALUE)
            if self.COLOUR == True:
                _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
                image.append(_img/255)
            else:
                image.append(_img/255)
            
        if self.augment == True:
            for num in range(0, len(image)):
                image.append(cv2.flip(image[num], 0))
                GRADE.append(GRADE[num])
           
        image = np.array(image)
        GRADE = np.array(GRADE)
        
        
        return image, GRADE
    def create_arg_data(self):
        X_train, Y_train = self.__getitem__()
        
        return X_train, Y_train

    
    def on_epoch_end(self):
        pass

    

    
