"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from imgaug import augmenters as iaa
from skimage import io

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Root directory of the project
ROOT_DIR = "/scratch/a.cot12/Mask_RCNN/samples/ultrasound/"

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_WEIGHTS_PATH)
SCRATCH_DIR = "/scratch/a.cot12/Mask_RCNN"
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(SCRATCH_DIR, "logs")

############################################################
#  Configurations
############################################################


class UltrasoundConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Ultrasound"
    BACKBONE = "resnet101"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + abnormality

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 739
    VALIDATION_STEPS = 185
    TOP_DOWN_PYRAMID_SIZE = 256

    # Skip detections with < 90% confidence
    IMAGE_MIN_DIM = 224


    LEARNING_RATE = 0.0005
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 1
    
    


############################################################
#  Dataset
############################################################

class UltrasoundDataset(utils.Dataset):

    def load_Ultrasound(self, dataset_dir, ids):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("abnormality", 1, "abnormality")
        
        # Train or validation dataset?
        #assert subset in ["DATASET1_TRAIN", "DATASET1_VAL", "DATASET1_TEST"]
        #image_ids = []
        
        dataset_dir = os.path.join(dataset_dir, "ALL_IMAGES")
        images_path = os.path.join(dataset_dir, "images")
        masks_path = os.path.join(dataset_dir, "masks")
        for name in ids:
            image_path = os.path.join(images_path, name + ".png")
            image = skimage.io.imread(image_path)
            mask_path = os.path.join(masks_path, name +".png")
            height = image.shape[0]
            width = image.shape[1]
            self.add_image(
                "abnormality",
                image_id=name,  # use file name as a unique image id
                path=image_path,
                mask_path = mask_path,
                width = width, height = height)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        info = self.image_info[image_id]
        mask_path =  info["mask_path"]
        
        mask = skimage.io.imread(mask_path, as_gray=True)
        mask_base = np.zeros((mask.shape[0],mask.shape[1], 1),dtype=np.uint8)
        for kk in range(0, mask.shape[0]):
            for ii in range(0, mask.shape[1]):
                if mask[kk,ii] > 0:
                    mask_base[kk,ii,0] = 1
        return mask_base.astype(np.bool), np.ones([mask_base.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ultrasound":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, fold):
    """Train the model."""
    # Training dataset.
    #image_size = 256
    ids_list = open(os.path.join("/scratch/a.cot12/Mask_RCNN/samples/ultrasound/dataset_lists",  "train_FD_{}".format(fold) + '.txt')).readlines()
    train_split  =int(round(len(ids_list)*0.8,0))
    ids_list = [x[:-1] for x in ids_list]
    print(ids_list)
    dataset_train = UltrasoundDataset()
    dataset_train.load_Ultrasound(args.dataset, ids_list[:train_split])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = UltrasoundDataset()
    dataset_val.load_Ultrasound(args.dataset, ids_list[train_split:])
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.2),
        iaa.Flipud(0.1),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    #print("Train network heads")
    #model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE,
                #epochs=100,
                #augmentation=augmentation,
                #layers='heads')

    print("Train all layers")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')




############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--fold', required=True,
                        metavar="fold",
                        help='The fold you wish to run')
    
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = UltrasoundConfig()
    else:
        class InferenceConfig(UltrasoundConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.fold)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
