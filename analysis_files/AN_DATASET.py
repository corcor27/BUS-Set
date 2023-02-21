import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import pandas as pd
import matplotlib.image as mpimg 
import scipy.ndimage as ndi 

def excel_reader(input_file, name):
    data = pd.read_excel(input_file)
    count_row = data.shape[0]
    for row in range(0, count_row):
        if str(data['Name'].iloc[row]) + ".png" == name:
            return data['Origin_dataset'].iloc[row]
    
def MASK_READ(path):
    read_image = cv2.imread(path,0)
    if read_image.shape[0] != 224:
        image1 = cv2.resize(read_image, (224,224))
    else:
        image1 = read_image
    scale_image = image1.astype(np.bool_)
    return scale_image

def DICE(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    array_sum = np.sum(y_true_f) + np.sum(y_pred_f)
    
    return (2. * intersection) / array_sum

def IOU(y_true, y_pred, NUM_CLASSES):
    con_matrix = np.zeros((2, 2, NUM_CLASSES))
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    for num in range(0, NUM_CLASSES):
        for pixel in range(0, len(y_true_f)):
            if y_true_f[pixel] == num:
                if y_pred_f[pixel] == num:
                   con_matrix[1, 1, num] += 1
                else:
                    con_matrix[0, 1, num] += 1
            else:
                if y_pred_f[pixel] == num:
                   con_matrix[1, 0, num] += 1
                else:
                    con_matrix[0, 0, num] += 1
    iou = ((con_matrix[1, 1, 0]/ (con_matrix[1, 1, 0] + con_matrix[0, 1, 0] + con_matrix[1, 0, 0])) + (con_matrix[1, 1, 1]/ (con_matrix[1, 1, 1] + con_matrix[0, 1, 1] + con_matrix[1, 0, 1])))/2
    acc = (((con_matrix[1, 1, 0] + con_matrix[0, 0, 0])/ ((con_matrix[1, 1, 0]) + con_matrix[0, 1, 0] + con_matrix[1, 0, 0] + con_matrix[0, 0, 0])))
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

def DATASET_CHOICE(DIR):
    if DIR == "BUSI":
        return 0
    elif DIR == "RODTOOK":
        return 1
    elif DIR == "OASBUD":
        return 2
    elif DIR == "UDIAT":
        return 3
    else:
        return 4
    
DATASET_OPTIONS = ["BUSI_TEST\masks", "RODTOOK_TEST\masks", "OASBUD_TEST\masks", "BUS_UDIAT_TEST\masks", "DATASET1_TEST\masks"]
EXCEL_OPTIONS = ["Excel_files\BUSI_TEST.xlsx", "Excel_files\RODTOOK_TEST.xlsx", "Excel_files\OASBUD_TEST.xlsx", "Excel_files\BUS_UDIAT_TEST.xlsx", "Excel_files\DATASET1_TEST.xlsx"]

BASE_PATH = r"D:\Documents\ultrasound_deeplearning\ULTRASOUND_DATASET"
#PREDICTIONS_PATH = r"D:\Documents\ultrasound_deeplearning\ULTRASOUND_DATASET\PREDICTIONS\ATTSK-UNET"
PREDICTIONS_PATH = r"D:\dropbox\joint_sets"
B_NAME = []
B_AV_DICE = []
B_AV_IoU = []
B_TP = []
B_FP = []
B_TPR = []
B_MID_DICE = []
B_DICE_MEDIAN = []
B_DICE_STD = []
B_UDICE_MEDIAN = []
B_UDICE_STD = []
B_IOU_MEDIAN = []
B_IOU_STD = []
B_AV_ACC = []
B_ACC_MEDIAN = []
B_ACC_STD = []

O_NAME = []
O_AV_DICE = []
O_AV_IoU = []
O_TP = []
O_FP = []
O_TPR = []
O_MID_DICE = []
O_DICE_MEDIAN = []
O_DICE_STD = []
O_UDICE_MEDIAN = []
O_UDICE_STD = []
O_IOU_MEDIAN = []
O_IOU_STD = []
O_AV_ACC = []
O_ACC_MEDIAN = []
O_ACC_STD = []

R_NAME = []
R_AV_DICE = []
R_AV_IoU = []
R_TP = []
R_FP = []
R_TPR = []
R_MID_DICE = []
R_DICE_MEDIAN = []
R_DICE_STD = []
R_UDICE_MEDIAN = []
R_UDICE_STD = []
R_IOU_MEDIAN = []
R_IOU_STD = []
R_AV_ACC = []
R_ACC_MEDIAN = []
R_ACC_STD = []

U_NAME = []
U_AV_DICE = []
U_AV_IoU = []
U_TP = []
U_FP = []
U_TPR = []
U_MID_DICE = []
U_DICE_MEDIAN = []
U_DICE_STD = []
U_UDICE_MEDIAN = []
U_UDICE_STD = []
U_IOU_MEDIAN = []
U_IOU_STD = []
U_AV_ACC = []
U_ACC_MEDIAN = []
U_ACC_STD = []

MODEL_DIR_LIST = os.listdir(PREDICTIONS_PATH)
for DIR in MODEL_DIR_LIST:
    DATA = DATASET_CHOICE(DIR)
    GT_PATH = os.path.join(BASE_PATH, DATASET_OPTIONS[DATA])
    PREDICT_MASK_PATH = os.path.join(PREDICTIONS_PATH, DIR)
    EXCEL_PATH = os.path.join(BASE_PATH, EXCEL_OPTIONS[DATA])
    B_dice = []
    B_iou = []
    B_tp = []
    B_fp = []
    B_mid_dice = []
    B_acc = []
    
    O_dice = []
    O_iou = []
    O_tp = []
    O_fp = []
    O_mid_dice = []
    O_acc = []
    
    R_dice = []
    R_iou = []
    R_tp = []
    R_fp = []
    R_mid_dice = []
    R_acc = []
    
    U_dice = []
    U_iou = []
    U_tp = []
    U_fp = []
    U_mid_dice = []
    U_acc = []
    
    MASK_LIST = os.listdir(PREDICT_MASK_PATH)
    for MASK in MASK_LIST:
        GT_MASK_PATH = os.path.join(GT_PATH, MASK)
        PRED_MASK_PATH = os.path.join(PREDICT_MASK_PATH, MASK)
        Type = str(excel_reader(EXCEL_PATH, MASK))
        GT_MASK = MASK_READ(GT_MASK_PATH)
        PRED_MASK = MASK_READ(PRED_MASK_PATH)
        DICE_VAL = DICE(GT_MASK, PRED_MASK)
        IOU_VAL = IOU(GT_MASK, PRED_MASK, 2)
        x,y,w,h = bounding_rectangle(GT_MASK)
        PRED_CENTRE = image_centre(PRED_MASK)
        ROI_CHECK = centre_check(x, y, x+w, y+h, PRED_CENTRE[0], PRED_CENTRE[1])
        if Type == "BUSI":
            B_dice.append(round(DICE_VAL,3))
            if round(DICE_VAL,3) > 0.500:
                B_mid_dice.append(round(DICE_VAL,3))
            B_iou.append(round(IOU_VAL[0],3))
            B_acc.append(round(IOU_VAL[1],3))
            BLANK_CHECK = np.sum(PRED_MASK)
            if BLANK_CHECK != 0:
                x,y,w,h = bounding_rectangle(GT_MASK)
                PRED_CENTRE = image_centre(PRED_MASK)
                ROI_CHECK = centre_check(x, y, x+w, y+h, PRED_CENTRE[0], PRED_CENTRE[1])
                if ROI_CHECK == 1:
                    B_tp.append(1)
                    B_fp.append(0)
                else:
                    B_tp.append(0)
                    B_fp.append(1)
            else:
                B_tp.append(0)
                B_fp.append(1)
                
        elif Type == "OASBUD":
            O_dice.append(round(DICE_VAL,3))
            if round(DICE_VAL,3) > 0.500:
                O_mid_dice.append(round(DICE_VAL,3))
            O_iou.append(round(IOU_VAL[0],3))
            O_acc.append(round(IOU_VAL[1],3))
            BLANK_CHECK = np.sum(PRED_MASK)
            if BLANK_CHECK != 0:
                x,y,w,h = bounding_rectangle(GT_MASK)
                PRED_CENTRE = image_centre(PRED_MASK)
                ROI_CHECK = centre_check(x, y, x+w, y+h, PRED_CENTRE[0], PRED_CENTRE[1])
                if ROI_CHECK == 1:
                    O_tp.append(1)
                    O_fp.append(0)
                else:
                    O_tp.append(0)
                    O_fp.append(1)
            else:
                O_tp.append(0)
                O_fp.append(1)
        
        elif Type == "RODTOOK":
            R_dice.append(round(DICE_VAL,3))
            if round(DICE_VAL,3) > 0.500:
                R_mid_dice.append(round(DICE_VAL,3))
            R_iou.append(round(IOU_VAL[0],3))
            R_acc.append(round(IOU_VAL[1],3))
            BLANK_CHECK = np.sum(PRED_MASK)
            if BLANK_CHECK != 0:
                x,y,w,h = bounding_rectangle(GT_MASK)
                PRED_CENTRE = image_centre(PRED_MASK)
                ROI_CHECK = centre_check(x, y, x+w, y+h, PRED_CENTRE[0], PRED_CENTRE[1])
                if ROI_CHECK == 1:
                    R_tp.append(1)
                    R_fp.append(0)
                else:
                    R_tp.append(0)
                    R_fp.append(1)
            else:
                R_tp.append(0)
                R_fp.append(1)
                
        elif Type == "BUS_UDIAT":
            U_dice.append(round(DICE_VAL,3))
            if round(DICE_VAL,3) > 0.500:
                U_mid_dice.append(round(DICE_VAL,3))
            U_iou.append(round(IOU_VAL[0],3))
            U_acc.append(round(IOU_VAL[1],3))
            BLANK_CHECK = np.sum(PRED_MASK)
            if BLANK_CHECK != 0:
                x,y,w,h = bounding_rectangle(GT_MASK)
                PRED_CENTRE = image_centre(PRED_MASK)
                ROI_CHECK = centre_check(x, y, x+w, y+h, PRED_CENTRE[0], PRED_CENTRE[1])
                if ROI_CHECK == 1:
                    U_tp.append(1)
                    U_fp.append(0)
                else:
                    U_tp.append(0)
                    U_fp.append(1)
            else:
                U_tp.append(0)
                U_fp.append(1)
        
        

        
    B_NAME.append(DIR)
    B_AV_DICE.append(round(sum(B_dice)/len(B_dice),3))
    B_AV_IoU.append(round(sum(B_iou)/len(B_iou),3))
    B_MID_DICE.append(round(sum(B_mid_dice)/len(B_mid_dice),3))
    B_TP.append(sum(B_tp))
    B_FP.append(sum(B_fp))
    B_TPR.append(round(sum(B_TP)/(sum(B_TP) + sum(B_FP)), 3))
    B_DICE_MEDIAN.append(round(np.median(B_dice),3))
    B_DICE_STD.append(round(np.std(B_dice),3))
    B_UDICE_MEDIAN.append(round(np.median(B_mid_dice), 3))
    B_UDICE_STD.append(round(np.std(B_mid_dice),3))
    B_IOU_MEDIAN.append(round(np.median(B_iou),3))
    B_IOU_STD.append(round(np.std(B_iou),3))
    B_AV_ACC.append(round(sum(B_acc)/len(B_acc),3))
    B_ACC_MEDIAN.append(round(np.median(B_acc),3))
    B_ACC_STD.append(round(np.std(B_acc),3))
    
    
    
    O_NAME.append(DIR)
    O_AV_DICE.append(round(sum(O_dice)/len(O_dice),3))
    O_AV_IoU.append(round(sum(O_iou)/len(O_iou),3))
    O_MID_DICE.append(round(sum(O_mid_dice)/len(O_mid_dice),3))
    O_TP.append(sum(O_tp))
    O_FP.append(sum(O_fp))
    O_TPR.append(round(sum(O_TP)/(sum(O_TP) + sum(O_FP)), 3))
    O_DICE_MEDIAN.append(round(np.median(O_dice),3))
    O_DICE_STD.append(round(np.std(O_dice),3))
    O_UDICE_MEDIAN.append(round(np.median(O_mid_dice), 3))
    O_UDICE_STD.append(round(np.std(O_mid_dice),3))
    O_IOU_MEDIAN.append(round(np.median(O_iou),3))
    O_IOU_STD.append(round(np.std(O_iou),3))
    O_AV_ACC.append(round(sum(O_acc)/len(O_acc),3))
    O_ACC_MEDIAN.append(round(np.median(O_acc),3))
    O_ACC_STD.append(round(np.std(O_acc),3))
    
    R_NAME.append(DIR)
    R_AV_DICE.append(round(sum(R_dice)/len(R_dice),3))
    R_AV_IoU.append(round(sum(R_iou)/len(R_iou),3))
    R_MID_DICE.append(round(sum(R_mid_dice)/len(R_mid_dice),3))
    R_TP.append(sum(R_tp))
    R_FP.append(sum(R_fp))
    R_TPR.append(round(sum(R_TP)/(sum(R_TP) + sum(R_FP)), 3))
    R_DICE_MEDIAN.append(round(np.median(R_dice),3))
    R_DICE_STD.append(round(np.std(R_dice),3))
    R_UDICE_MEDIAN.append(round(np.median(R_mid_dice), 3))
    R_UDICE_STD.append(round(np.std(R_mid_dice),3))
    R_IOU_MEDIAN.append(round(np.median(R_iou),3))
    R_IOU_STD.append(round(np.std(R_iou),3))
    R_AV_ACC.append(round(sum(R_acc)/len(R_acc),3))
    R_ACC_MEDIAN.append(round(np.median(R_acc),3))
    R_ACC_STD.append(round(np.std(R_acc),3))
    
    U_NAME.append(DIR)
    U_AV_DICE.append(round(sum(U_dice)/len(U_dice),3))
    U_AV_IoU.append(round(sum(U_iou)/len(U_iou),3))
    U_MID_DICE.append(round(sum(U_mid_dice)/len(U_mid_dice),3))
    U_TP.append(sum(U_tp))
    U_FP.append(sum(U_fp))
    U_TPR.append(round(sum(U_TP)/(sum(U_TP) + sum(U_FP)), 3))
    U_DICE_MEDIAN.append(round(np.median(U_dice),3))
    U_DICE_STD.append(round(np.std(U_dice),3))
    U_UDICE_MEDIAN.append(round(np.median(U_mid_dice), 3))
    U_UDICE_STD.append(round(np.std(U_mid_dice),3))
    U_IOU_MEDIAN.append(round(np.median(U_iou),3))
    U_IOU_STD.append(round(np.std(U_iou),3))
    U_AV_ACC.append(round(sum(U_acc)/len(U_acc),3))
    U_ACC_MEDIAN.append(round(np.median(U_acc),3))
    U_ACC_STD.append(round(np.std(U_acc),3))
    
    
        
for ii in range(0, len(B_NAME)):
    print("BUSI")
    print("NAME: {} ".format(B_NAME[ii]))
    print("DICE: {} Median: {} std: {}".format(B_AV_DICE[ii], B_DICE_MEDIAN[ii], B_DICE_STD[ii]))
    print("DICE(>0.5): {} Median: {} std: {}".format(B_MID_DICE[ii], B_UDICE_MEDIAN[ii], B_UDICE_STD[ii]))
    print("IoU: {} Median: {} std: {}".format(B_AV_IoU[ii], B_IOU_MEDIAN[ii], B_IOU_STD[ii]))
    print("ACC: {} Median: {} std: {}".format(B_AV_ACC[ii], B_ACC_MEDIAN[ii], B_ACC_STD[ii]))
    print("TP: {} FP: {} TPR: {}".format(B_TP[ii], B_FP[ii], B_TPR[ii]))    
    
for ii in range(0, len(O_NAME)):
    print("OASBUD")
    print("NAME: {} ".format(O_NAME[ii]))
    print("DICE: {} Median: {} std: {}".format(O_AV_DICE[ii], O_DICE_MEDIAN[ii], O_DICE_STD[ii]))
    print("DICE(>0.5): {} Median: {} std: {}".format(O_MID_DICE[ii], O_UDICE_MEDIAN[ii], O_UDICE_STD[ii]))
    print("IoU: {} Median: {} std: {}".format(O_AV_IoU[ii], O_IOU_MEDIAN[ii], O_IOU_STD[ii]))
    print("ACC: {} Median: {} std: {}".format(O_AV_ACC[ii], O_ACC_MEDIAN[ii], O_ACC_STD[ii]))
    print("TP: {} FP: {} TPR: {}".format(O_TP[ii], O_FP[ii], O_TPR[ii]))    
    
for ii in range(0, len(R_NAME)):
    print("RODTOOK")
    print("NAME: {} ".format(R_NAME[ii]))
    print("DICE: {} Median: {} std: {}".format(R_AV_DICE[ii], R_DICE_MEDIAN[ii], R_DICE_STD[ii]))
    print("DICE(>0.5): {} Median: {} std: {}".format(R_MID_DICE[ii], R_UDICE_MEDIAN[ii], R_UDICE_STD[ii]))
    print("IoU: {} Median: {} std: {}".format(R_AV_IoU[ii], R_IOU_MEDIAN[ii], R_IOU_STD[ii]))
    print("ACC: {} Median: {} std: {}".format(R_AV_ACC[ii], R_ACC_MEDIAN[ii], R_ACC_STD[ii]))
    print("TP: {} FP: {} TPR: {}".format(R_TP[ii], R_FP[ii], R_TPR[ii]))    
    
for ii in range(0, len(U_NAME)):
    print("UDIAT")
    print("NAME: {} ".format(U_NAME[ii]))
    print("DICE: {} Median: {} std: {}".format(U_AV_DICE[ii], U_DICE_MEDIAN[ii], U_DICE_STD[ii]))
    print("DICE(>0.5): {} Median: {} std: {}".format(U_MID_DICE[ii], U_UDICE_MEDIAN[ii], U_UDICE_STD[ii]))
    print("IoU: {} Median: {} std: {}".format(U_AV_IoU[ii], U_IOU_MEDIAN[ii], U_IOU_STD[ii]))
    print("ACC: {} Median: {} std: {}".format(U_AV_ACC[ii], U_ACC_MEDIAN[ii], U_ACC_STD[ii]))
    print("TP: {} FP: {} TPR: {}".format(U_TP[ii], U_FP[ii], U_TPR[ii]))    



