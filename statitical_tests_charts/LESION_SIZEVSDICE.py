import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import pandas as pd
import matplotlib.image as mpimg 
import scipy.ndimage as ndi 
from scipy import ndimage

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

def LESION_SIZE_PIXEL(MASK):
    SIZE = np.sum(MASK)
    return SIZE

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
    return iou

def excel_reader(input_file, name):
    data = pd.read_excel(input_file)
    count_row = data.shape[0]
    for row in range(0, count_row):
        if str(data['Name'].iloc[row]) + ".png" == name:
            return int(data['Class'].iloc[row])
        
MODELS = ["MaskRCNN", "SK-U-Net", "TransUNet"]
MODEL_PATHS = ["MASK", "SK", "TRAN"]
BASE_PATH = "ALL_IMAGES/masks"
PREDICTIONS_PATH = "PRED"
EXCEL_PATH = "ALL_IMAGES.xlsx"

data = pd.read_excel(EXCEL_PATH)

IMAGE_LIST = list(data["Name"])
GT_SIZES = []
NAMES = []
GROUP = []
DATASET = []
for IMG in range(0, len(IMAGE_LIST)):
    GT_MASK_PATH = os.path.join(BASE_PATH, IMAGE_LIST[IMG] + ".png")
    GT_MASK = MASK_READ(GT_MASK_PATH)
    MASK_LESION_SIZE = LESION_SIZE_PIXEL(GT_MASK)
    GT_SIZES.append(MASK_LESION_SIZE)
    NAMES.append(IMAGE_LIST[IMG])
    DATASET.append(data["Origin_dataset"].iloc[IMG])
GT_SIZES = np.array(GT_SIZES)   
q3, q1 = np.percentile(GT_SIZES, [75 ,25])
check_vals = [q1, np.median(GT_SIZES), q3]

for ii in range(0, len(NAMES)):
    if GT_SIZES[ii] < check_vals[0]:
        GROUP.append(0)
    elif check_vals[0] <= GT_SIZES[ii] < check_vals[1]:
        GROUP.append(1)
    elif check_vals[1] <= GT_SIZES[ii] < check_vals[2]:
        GROUP.append(2)
    elif GT_SIZES[ii] >= check_vals[2]:
        GROUP.append(3)

GROUP = np.array(GROUP)
DATASET= np.array(DATASET)

df = pd.DataFrame()

df["Names"] = NAMES
df["GT_SIZES"] = GT_SIZES
df["GROUP"] = GROUP
df["Dataset"] = DATASET
factor = 10

for model in range(0, len(MODELS)):
    SIZE_LIST = []
    DICE_LIST = []
    #IMAGE_LIST = os.listdir(BASE_PATH)
    for IMG in IMAGE_LIST:
        GT_MASK_PATH = os.path.join(BASE_PATH, IMG + ".png")
        GT_MASK = MASK_READ(GT_MASK_PATH)
        PRED_MASK_PATH = os.path.join(PREDICTIONS_PATH, MODEL_PATHS[model], IMG + ".png")
        PRED_MASK = MASK_READ(PRED_MASK_PATH)
        LESION_SIZE = LESION_SIZE_PIXEL(PRED_MASK)
        #DICE_VAL = DICE(GT_MASK, PRED_MASK)
        LE = LESION_SIZE_PIXEL(GT_MASK)
        
        SIZE_LIST.append(LESION_SIZE)
        DICE_LIST.append(LE)
    df["{}_SIZES".format(MODEL_PATHS[model])] = SIZE_LIST
    df["{}_DICES".format(MODEL_PATHS[model])] = DICE_LIST
for ii in range(0, len(MODEL_PATHS)):
    
    df0 = df[df.GROUP  == 0]
    df1 = df[df.GROUP  == 1]
    df2 = df[df.GROUP  == 2]
    df3 = df[df.GROUP  == 3]
    
    MASK0 = np.array(df0["{}_SIZES".format(MODEL_PATHS[ii])])
    MASK1 = np.array(df1["{}_SIZES".format(MODEL_PATHS[ii])])
    MASK2 = np.array(df2["{}_SIZES".format(MODEL_PATHS[ii])])
    MASK3 = np.array(df3["{}_SIZES".format(MODEL_PATHS[ii])])
    
    #GT0 = np.array(df0["GT_SIZES"])
    #GT1 = np.array(df1["GT_SIZES"])
    #GT2 = np.array(df2["GT_SIZES"])
    #GT3 = np.array(df3["GT_SIZES"])
       
    
    GT0 = np.array(df0["{}_DICES".format(MODEL_PATHS[ii])])
    GT1 = np.array(df1["{}_DICES".format(MODEL_PATHS[ii])])
    GT2 = np.array(df2["{}_DICES".format(MODEL_PATHS[ii])])
    GT3 = np.array(df3["{}_DICES".format(MODEL_PATHS[ii])])
    #ndimage.measurements.center_of_mass(a)
    print(GT0.shape, MASK0.shape)
    plt.scatter(GT0, MASK0 ,color='b', label='Size < 25%' , s=5)
    plt.scatter(GT1, MASK1 ,color='r', label='25% =< Size < 50%', s=5 )
    plt.scatter(GT2, MASK2 ,color='g', label='50% =< Size < 75%', s=5 )
    plt.scatter(GT3, MASK3 ,color='k', label=' Size => 75%', s=5 )
    plt.legend(loc = "upper right")
    plt.axline((0, 0), slope=1)
    plt.title("{}".format(MODELS[ii]))
    #plt.xlabel("DCS Score")
    plt.xlabel("Ground Truth Pixel Size")
    plt.ylabel("{} Prediction Pixel Size".format(MODELS[ii]))
    output = "CHARTS/{}_TYPE_{}.png".format(MODEL_PATHS[ii], "DICE2")
    plt.xlim([0, 30000])
    plt.ylim([0, 30000])
    plt.savefig(output)
    plt.close()


    
"""
mG0 = np.mean(GT0)
    mP0 = np.mean(MASK0)
    mG1 = np.mean(GT1)
    mP1 = np.mean(MASK1)
    mG2 = np.mean(GT2)
    mP2 = np.mean(MASK2)
    mG3 = np.mean(GT3)
    mP3 = np.mean(MASK3)
    print(MODELS[ii])
    print("Size < 25%", "GT {} {}".format(round(np.mean(GT0),2), round(np.std(GT0), 2)), "Pred {} {}".format(round(np.mean(MASK0),2), round(np.std(MASK0),2)))
    print("25% =< Size < 50%", "GT {} {}".format(round(np.mean(GT1),2), round(np.std(GT1), 2)), "Pred {} {}".format(round(np.mean(MASK1),2), round(np.std(MASK1),2)))
    print("50% =< Size < 75%", "GT {} {}".format(round(np.mean(GT2),2), round(np.std(GT2), 2)), "Pred {} {}".format(round(np.mean(MASK2),2), round(np.std(MASK2),2)))
    print("Size => 75%", "GT {} {}".format(round(np.mean(GT3),2), round(np.std(GT3), 2)), "Pred {} {}".format(round(np.mean(MASK3),2), round(np.std(MASK3),2))) 
MODELS = ["MaskRCNN", "SK-U-Net", "TransUNet"]
MODEL_PATHS = ["MASK", "SK", "TRAN"]
BASE_PATH = "ALL_IMAGES/masks"
PREDICTIONS_PATH = "PRED"
EXCEL_PATH = "ALL_IMAGES.xlsx"


for model in range(0, len(MODELS)):
    B_DICE_LIST = []
    B_SIZE_LIST = []
    B_IOU_LIST = []
    M_DICE_LIST = []
    M_SIZE_LIST = []
    M_IOU_LIST = []
    
    IMAGE_LIST = os.listdir(BASE_PATH)
    for IMG in IMAGE_LIST:
        GT_MASK_PATH = os.path.join(BASE_PATH, IMG)
        PRED_MASK_PATH = os.path.join(PREDICTIONS_PATH, MODEL_PATHS[model], IMG)
        GT_MASK = MASK_READ(GT_MASK_PATH)
        PRED_MASK = MASK_READ(PRED_MASK_PATH)
        DICE_VAL = DICE(GT_MASK, PRED_MASK)
        IOU_VAL = IOU(GT_MASK, PRED_MASK, 2)
        LESION_SIZE = LESION_SIZE_PIXEL(GT_MASK)
        TYPE = excel_reader(EXCEL_PATH, IMG)

        if TYPE == 0:
            B_DICE_LIST.append(DICE_VAL)
            B_SIZE_LIST.append(LESION_SIZE)
            B_IOU_LIST.append(IOU_VAL)
        elif TYPE == 1:
            M_DICE_LIST.append(DICE_VAL)
            M_SIZE_LIST.append(LESION_SIZE)
            M_IOU_LIST.append(IOU_VAL)
        
        DICE_LIST = np.array(B_DICE_LIST + M_DICE_LIST )
        SIZE_LIST = np.array(B_SIZE_LIST + M_SIZE_LIST )

        #plt.scatter(B_SIZE_LIST, B_DICE_LIST,color='b', label='Benign' )
        #plt.scatter(M_SIZE_LIST, M_DICE_LIST, color='r', label='Malignant')
        #plt.plot(np.unique(SIZE_LIST), np.poly1d(np.polyfit(SIZE_LIST, DICE_LIST, 1))(np.unique(SIZE_LIST)))
        #plt.xlabel("Pixel size")
        #plt.ylabel("DSC")
        #plt.title(MODELS[model])
        #plt.legend()
        #output = "CHARTS/{}_TYPE_{}.png".format(MODELS[model], "DICE")
        #plt.savefig(output)
        #plt.close()
"""   