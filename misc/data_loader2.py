import os
import sys
import random
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

from tensorflow.keras.applications.resnet import preprocess_input



class DataGen(keras.utils.Sequence):
    def __init__(self,ids, img_path, mask_path, image_size, augment, resnet_preproc=False):
        self.img_path = img_path
        self.ids = ids
        self.mask_path = mask_path
        self.image_size = image_size
        self.augment = augment
        self.resnet_preproc = resnet_preproc
        self.on_epoch_end()
        


        
    def __len__(self):
        return int(len(self.ids))
        
    def __load__(self, NAME):
        #Path
        IMG_PATH = os.path.join(self.img_path,"{}.png".format(NAME))
        #print(IMG_PATH)
        MASK_PATH = os.path.join(self.mask_path, "{}.png".format(NAME))
        #Read Image
        image = cv2.imread(IMG_PATH)
        mask = cv2.imread(MASK_PATH, 0)
        #Read Masks
        #Normalizaing
        
        if self.resnet_preproc==True:
            
            image = preprocess_input(image[np.newaxis, ...]).squeeze()
            
        else:
            
            image = image/255
            
        if self.image_size!=image.shape[0]:

            image = cv2.resize(image, (self.image_size, self.image_size), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST)
            
        mask[mask >= 1] = 1
        
        # mask = mask//255
        
        mask = mask.astype(np.float32)
        
        return image, mask

    def __getitem__(self):
        
        image = []
        mask = []
        
        for NAME in self.ids:
            img, mas = self.__load__(NAME)
            image.append(img)
            mask.append(mas)
        if self.augment == True:
            for num in range(0, len(image)):
                image.append(cv2.flip(image[num], 0))
                mask.append(cv2.flip(mask[num], 0))
           
        image = np.array(image)
        mask = np.array(mask)
        
        
        return image, mask
    def create_arg_data(self):
        X_train, Y_train = self.__getitem__()
        
        return X_train, Y_train

    
    def on_epoch_end(self):
        pass

    

    
