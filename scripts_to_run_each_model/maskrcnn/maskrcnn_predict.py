import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from tensorflow import keras
from skimage import io

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/a.cot12/Mask_RCNN/samples/ultrasound/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.ultrasound import ultrasound


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = ultrasound.UltrasoundConfig()
Ultrasound_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

TEST_MODE = "inference"

dataset = ultrasound.UltrasoundDataset()
dataset.load_Ultrasound(Ultrasound_DIR, "val")

dataset.prepare()


with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = model.find_last()

# Load weights
model.load_weights(weights_path, by_name=True)
image_numbers = dataset.image_ids

image_info = dataset.image_info[image_numbers[0]]
fname = image_info["id"]
image = dataset.load_image(image_numbers[0])
mask = dataset.load_mask(image_numbers[0])[0]
mask_image = np.squeeze(mask)
output = "/home/a.cot12/Mask_RCNN/samples/ultrasound/predictions/%s.png" %(fname.replace('.png',''))
print(fname)
cv2.imwrite(output, mask_image)
"""
for i in dataset.image_ids:
    print(dataset.image_ids(i))
    image = dataset.load_image(i)
    output = "/home/a.cot12/Mask_RCNN/samples/ultrasound/predictions/%s.png" %(dataset.image_ids[i])
    cv2.imwrite(output, image)
    image_list = []
    image_list.append(image)
    image_base = np.array(image_list)
    results = model.detect(image_base, verbose=0)[0]
    masks = results['masks']
    for i in range(0,masks.shape[2]):
        masksingle = np.squeeze(masks[:,:,i])
        output = "/home/a.cot12/Mask_RCNN/samples/ultrasound/predictions/%s.png" %(image_ids[i])
        cv2.imwrite(output, masksingle)
    print(results)
    print("&&&&&&&&&&&: "+str(results['rois']))
    print("&&&&&&&&&&&: "+str(results['class_ids']))
    print("&&&&&&&&&&&: "+str(results['scores']))
"""
