import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import pandas as pd
import matplotlib.image as mpimg 
import scipy.ndimage as ndi 
import pandas

import cv2
import numpy as np

import os
import scipy.ndimage as ndi 


"""
def MASK_READ(path):
    read_image = cv2.imread(path,0)
    if read_image.shape[0] != 224:
        image1 = cv2.resize(read_image, (224,224))
    else:
        image1 = read_image
    scale_image = image1.astype(np.bool_)
    return scale_image

"""
def MASK_READ(path):
    read_image = cv2.imread(path,0)
    if read_image.shape[0] != 224:
        image1 = cv2.resize(read_image, (224,224))
    else:
        image1 = read_image
    image1 = image1/255
    #scale_image = image1.astype(np.bool_)
    scale_image = image1.astype(np.float32)
    return scale_image

def DICE(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    array_sum = np.sum(y_true_f) + np.sum(y_pred_f)
    
    return (2. * intersection) / array_sum
    
def IOU(y_true, y_pred):
    con_matrix = np.zeros((2, 2))
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    for pixel in range(0, len(y_true_f)):
        if y_true_f[pixel] == 1:
            if y_pred_f[pixel] == 1:
                con_matrix[1, 1] += 1
            else:
                con_matrix[0, 1] += 1
        else:
            if y_pred_f[pixel] == 1:
                con_matrix[1, 0] += 1
            else:
                con_matrix[0, 0] += 1
    iou = ((con_matrix[1, 1]/ (con_matrix[1, 1] + con_matrix[0, 1] + con_matrix[1, 0])) )#+ (con_matrix[1, 1, 1]/ (con_matrix[1, 1, 1] + con_matrix[0, 1, 1] + con_matrix[1, 0, 1])))/2
    acc = (((con_matrix[1, 1] + con_matrix[0, 0])/ ((con_matrix[1, 1]) + con_matrix[0, 1] + con_matrix[1, 0] + con_matrix[0, 0])))
    return iou, acc
    


def bounding_rectangle(img):
    imgray = img.astype(np.uint8)
    ret,thresh = cv2.threshold(imgray,0,1,0)
    contour, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    x,y,w,h = cv2.boundingRect(contour[0])
    #bound = cv2.rectangle(imgray,(x,y),(x+w,y+h),(255,255,255),2)
    return x,y,w,h

def image_centre(image):
    imgray = image.astype(np.uint8)
    cy, cx = ndi.center_of_mass(imgray)
    roundcy = round(cy,0)
    roundcx = round(cx,0)
    return roundcx,roundcy    

def centre_check(minx, miny, maxx, maxy, cx, cy):
    if minx <= cx <= maxx:
        if miny <= cy <= maxy:
            return 1
        else: 
            return 0
    else:
        return 0
    if minx <= cx <= maxx:
        if miny <= cy <= maxy:
            return 1
        else: 
            return 0
    else:
        return 0



BASE_PATH = r"D:\deeplabv3\DICE_CHARTS_PREDICTION\PRED"
ground_path = r"D:\deeplabv3\ALL_IMAGES\masks"
EXCEL_PATH = r"D:\deeplabv3\Excel_files\ALL_IMAGES.xlsx"

data = pd.read_excel(EXCEL_PATH)
CLASS_LIST = list(data["Class"])
NAMES_LIST = list(data["Name"])
Dlist = list(data["Origin_dataset"])

MODEL_DIR_LIST = os.listdir(BASE_PATH)
#MODEL_DIR_LIST = ["MASK"]
for DIR in MODEL_DIR_LIST:
    print(DIR)
    Bm_dice = []
    Bm_iou = []
    Bm_tp = []
    Bm_fp = []
    Bm_mid_dice = []
    Bm_acc = []
    Bb_dice = []
    Bb_iou = []
    Bb_tp = []
    Bb_fp = []
    Bb_mid_dice = []
    Bb_acc = []
    Om_dice = []
    Om_iou = []
    Om_tp = []
    Om_fp = []
    Om_mid_dice = []
    Om_acc = []
    Ob_dice = []
    Ob_iou = []
    Ob_tp = []
    Ob_fp = []
    Ob_mid_dice = []
    Ob_acc = []
    Rm_dice = []
    Rm_iou = []
    Rm_tp = []
    Rm_fp = []
    Rm_mid_dice = []
    Rm_acc = []
    Rb_dice = []
    Rb_iou = []
    Rb_tp = []
    Rb_fp = []
    Rb_mid_dice = []
    Rb_acc = []
    Um_dice = []
    Um_iou = []
    Um_tp = []
    Um_fp = []
    Um_mid_dice = []
    Um_acc = []
    Ub_dice = []
    Ub_iou = []
    Ub_tp = []
    Ub_fp = []
    Ub_mid_dice = []
    Ub_acc = []
    PREDICT_MASK_PATH = os.path.join(BASE_PATH, DIR)
        
    MASK_LIST = os.listdir(PREDICT_MASK_PATH)
    for MASK in MASK_LIST:
        GT_MASK_PATH = os.path.join(PREDICT_MASK_PATH, MASK)
        PRED_MASK_PATH = os.path.join(ground_path, MASK)
        POS = NAMES_LIST.index(MASK.replace(".png", ""))
        Type = int(CLASS_LIST[POS])
        Dset = str(Dlist[POS])
        GT_MASK = MASK_READ(GT_MASK_PATH)
        PRED_MASK = MASK_READ(PRED_MASK_PATH)
        DICE_VAL = DICE(GT_MASK, PRED_MASK)
        IOU_VAL = IOU(GT_MASK, PRED_MASK)
        if Type == 1:
            if Dset == "BUSI":
                Bm_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Bm_mid_dice.append(DICE_VAL)
                Bm_iou.append(IOU_VAL[0])
                Bm_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Bm_tp.append(1)
                        Bm_fp.append(0)
                    else:
                        Bm_tp.append(0)
                        Bm_fp.append(1)
                else:
                    Bm_tp.append(0)
                    Bm_fp.append(1)
                            
            elif Dset == "BUS_UDIAT":
                Um_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Um_mid_dice.append(DICE_VAL)
                Um_iou.append(IOU_VAL[0])
                Um_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Um_tp.append(1)
                        Um_fp.append(0)
                    else:
                        Um_tp.append(0)
                        Um_fp.append(1)
                else:
                    Um_tp.append(0)
                    Um_fp.append(1)
                            
            elif Dset == "OASBUD":
                Om_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Om_mid_dice.append(DICE_VAL)
                Om_iou.append(IOU_VAL[0])
                Om_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Om_tp.append(1)
                        Om_fp.append(0)
                    else:
                        Om_tp.append(0)
                        Om_fp.append(1)
                else:
                    Om_tp.append(0)
                    Om_fp.append(1)
                            
            elif Dset == "RODTOOK":
                Rm_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Rm_mid_dice.append(DICE_VAL)
                Rm_iou.append(IOU_VAL[0])
                Rm_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Rm_tp.append(1)
                        Rm_fp.append(0)
                    else:
                        Rm_tp.append(0)
                        Rm_fp.append(1)
                else:
                    Rm_tp.append(0)
                    Rm_fp.append(1)
        else:
            if Dset == "BUSI":
                Bb_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Bb_mid_dice.append(DICE_VAL)
                Bb_iou.append(IOU_VAL[0])
                Bb_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Bb_tp.append(1)
                        Bb_fp.append(0)
                    else:
                        Bb_tp.append(0)
                        Bb_fp.append(1)
                else:
                    Bb_tp.append(0)
                    Bb_fp.append(1)
            elif Dset == "BUS_UDIAT":
                Ub_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Ub_mid_dice.append(DICE_VAL)
                Ub_iou.append(IOU_VAL[0])
                Ub_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Ub_tp.append(1)
                        Ub_fp.append(0)
                    else:
                        Ub_tp.append(0)
                        Ub_fp.append(1)
                else:
                    Ub_tp.append(0)
                    Ub_fp.append(1)
            elif Dset == "OASBUD":
                Ob_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Ob_mid_dice.append(DICE_VAL)
                Ob_iou.append(IOU_VAL[0])
                Ob_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Ob_tp.append(1)
                        Ob_fp.append(0)
                    else:
                        Ob_tp.append(0)
                        Ob_fp.append(1)
                else:
                    Ob_tp.append(0)
                    Ob_fp.append(1)
            elif Dset == "RODTOOK":
                Rb_dice.append(DICE_VAL)
                if round(DICE_VAL,3) >= 0.500:
                    Rb_mid_dice.append(DICE_VAL)
                Rb_iou.append(IOU_VAL[0])
                Rb_acc.append(IOU_VAL[1])
                        
                BLANK_CHECK = np.sum(PRED_MASK)
                if BLANK_CHECK != 0:
                    if round(DICE_VAL,3) >= 0.500:
                        Rb_tp.append(1)
                        Rb_fp.append(0)
                    else:
                        Rb_tp.append(0)
                        Rb_fp.append(1)
                else:
                    Rb_tp.append(0)
                    Rb_fp.append(1)

    print("MAG")
    print("BUSI")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Bm_dice),3), round(np.median(Bm_dice),3), round(np.std(Bm_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Bm_mid_dice),3), round(np.median(Bm_mid_dice), 3), round(np.std(Bm_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Bm_iou),3), round(np.median(Bm_iou),3), round(np.std(Bm_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Bm_tp), sum(Bm_fp), round(sum(Bm_tp)/(sum(Bm_tp) + sum(Bm_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Bm_acc),3), round(np.median(Bm_acc),3), round(np.std(Bm_acc),3)))
        

    print("BEN")
    print("BUSI")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Bb_dice),3), round(np.median(Bb_dice),3), round(np.std(Bb_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Bb_mid_dice),3), round(np.median(Bb_mid_dice), 3), round(np.std(Bb_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Bb_iou),3), round(np.median(Bb_iou),3), round(np.std(Bb_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Bb_tp), sum(Bb_fp), round(sum(Bb_tp)/(sum(Bb_tp) + sum(Bb_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Bb_acc),3), round(np.median(Bb_acc),3), round(np.std(Bb_acc),3)))
        

    
        
    print("MAG")
    print("OASBUD")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Om_dice),3), round(np.median(Om_dice),3), round(np.std(Om_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Om_mid_dice),3), round(np.median(Om_mid_dice), 3), round(np.std(Om_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Om_iou),3), round(np.median(Om_iou),3), round(np.std(Om_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Om_tp), sum(Om_fp), round(sum(Om_tp)/(sum(Om_tp) + sum(Om_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Om_acc),3), round(np.median(Om_acc),3), round(np.std(Om_acc),3)))
    

    print("BEN")
    print("OASBUD")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Ob_dice),3), round(np.median(Ob_dice),3), round(np.std(Ob_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Ob_mid_dice),3), round(np.median(Ob_mid_dice), 3), round(np.std(Ob_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Ob_iou),3), round(np.median(Ob_iou),3), round(np.std(Ob_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Ob_tp), sum(Ob_fp), round(sum(Ob_tp)/(sum(Ob_tp) + sum(Ob_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Ob_acc),3), round(np.median(Ob_acc),3), round(np.std(Ob_acc),3)))
        

    print("MAG")
    print("RODTOOK")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Rm_dice),3), round(np.median(Rm_dice),3), round(np.std(Rm_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Rm_mid_dice),3), round(np.median(Rm_mid_dice), 3), round(np.std(Rm_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Rm_iou),3), round(np.median(Rm_iou),3), round(np.std(Rm_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Rm_tp), sum(Rm_fp), round(sum(Rm_tp)/(sum(Rm_tp) + sum(Rm_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Rm_acc),3), round(np.median(Rm_acc),3), round(np.std(Rm_acc),3)))
        
    
    print("BEN")
    print("RODTOOK")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Rb_dice),3), round(np.median(Rb_dice),3), round(np.std(Rb_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Rb_mid_dice),3), round(np.median(Rb_mid_dice), 3), round(np.std(Rb_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Rb_iou),3), round(np.median(Rb_iou),3), round(np.std(Rb_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Rb_tp), sum(Rb_fp), round(sum(Rb_tp)/(sum(Rb_tp) + sum(Rb_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Rb_acc),3), round(np.median(Rb_acc),3), round(np.std(Rb_acc),3)))

    
    print("MAG")
    print("BUS_UDIAT")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Um_dice),3), round(np.median(Um_dice),3), round(np.std(Um_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Um_mid_dice),3), round(np.median(Um_mid_dice), 3), round(np.std(Um_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Um_iou),3), round(np.median(Um_iou),3), round(np.std(Um_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Um_tp), sum(Um_fp), round(sum(Um_tp)/(sum(Um_tp) + sum(Um_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Um_acc),3), round(np.median(Um_acc),3), round(np.std(Um_acc),3)))
        

    print("BEN")
    print("BUS_UDIAT")
    print("DICE: {} Median: {} std: {}".format(round(np.mean(Ub_dice),3), round(np.median(Ub_dice),3), round(np.std(Ub_dice),3)))
    print("DICE(>0.5): {} Median: {} std: {}".format(round(np.mean(Ub_mid_dice),3), round(np.median(Ub_mid_dice), 3), round(np.std(Ub_mid_dice),3)))
    print("IoU: {} Median: {} std: {}".format(round(np.mean(Ub_iou),3), round(np.median(Ub_iou),3), round(np.std(Ub_iou),3)))
    print("TP: {} FP: {} TPR: {}".format(sum(Ub_tp), sum(Ub_fp), round(sum(Ub_tp)/(sum(Ub_tp) + sum(Ub_fp)), 3)))
    print("F: {} median: {} std: {}".format(round(np.mean(Ub_acc),3), round(np.median(Ub_acc),3), round(np.std(Ub_acc),3)))
    






