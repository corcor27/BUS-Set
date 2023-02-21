import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndi 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import json
import ast
import pandas as pd
import scipy.stats as stats
import pandas as pd
import scipy.stats as stats
import pingouin as pg
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.stats.multitest

opt = [ "DEEP", "MASK", "U", "SK", "DENSE", "ATT", "PP", "SWIN", "TRAN"] #
names = ["DeepLabv3+", "Mask R-CNN", "U-Net", "Sk-U-Net", "Att-D-U-Net", "Att-U-Net", "U-Net++", "Swin-U-Net", "Trans-U-Net"]    


f = open('dic_auc_folds.txt', 'r')
f1 = json.loads(f.read())
df = pd.DataFrame.from_dict(f1)
print(df)
fvalue, pvalue = stats.f_oneway(df['ATT'], df['DEEP'], df['DENSE'], df['MASK'], df['PP'], df['SK'], df['SWIN'], df['TRAN'], df['U'])
print(fvalue, pvalue)    

values = []
models = []

cols = df.columns

for col in cols:
    for row in range(0, df.shape[0]):
        values.append(df[col].iloc[row])
        models.append(col)
dframe = pd.DataFrame()
dframe["mod"] = models
dframe["values"] = values
#print(dframe)
tukety = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=dframe['values'], groups=dframe['mod'], alpha=0.01)
print(tukety)


"""
def load_imgs(dic, mod):
    imgs = []
    
    img_list = list(dic["Name"])
    for item in img_list:
        P2 = os.path.join(mod, item + ".npy")
        img = np.load(P2)
        if img.shape == (224,224,0):

            img = np.zeros((img.shape[0], img.shape[1]))
        #print(np.min(img), np.max(img))
        imgs.append(img)
        #break
        
    imgs = np.array(imgs)
    return imgs

def load_masks(dic):
    masks = []
    img_list = list(dic["Name"])
    for item in img_list:
        #print(item)
        P2 = os.path.join("ALL_IMAGES/masks", item + ".png")
        mask = cv2.imread(P2, 0)
        #mask = mask//255image1 = image1/255
        mask = mask/255
        mask[mask >= 1] = 1
        mask = mask.astype(np.uint8)
        
        #mask[mask < 0.5] = 0
        #print(np.min(mask), np.max(mask))
        
        masks.append(mask)
        #break
        
    masks = np.array(masks)
    return masks

opt = [ "DEEP", "MASK", "U", "SK", "DENSE", "ATT", "PP", "SWIN", "TRAN"] #
names = ["Deeplab v3+", "MaskRCNN", "U-Net", "SK-U-Net", "ATT-D-U-Net", "ATT-U-Net", "UNet ++", "Swin-U-Net", "Trans-U-Net"]
#num_length = os.listdir()


#dic_tpr = {}
#dic_fpr = {}
dic_auc = {}
for ii in range(0, len(opt)):
    tprs = []
    aucs = []
    for kk in range(1, 6):
        EXCEL_PATH = "Excel_files/FOLD_{}.xlsx".format(kk)
        data = pd.read_excel(EXCEL_PATH)
        MASKS = load_masks(data)
        ground_truth_labels = MASKS.ravel()
        IMGS = load_imgs(data, opt[ii])
        #print(MASKS.shape, IMGS.shape)
        score_value = IMGS.ravel()
        fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_auc = auc(fpr,tpr)
        print(interp_tpr.shape)
        tprs.append(interp_tpr)
        
        aucs.append(roc_auc)
        
        #auc_list.append(roc_auc)
    #dic_auc[opt[ii]] = auc_list
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #dic_auc[opt[ii]] = [mean_auc, std_auc]
    dic_auc[opt[ii]] = aucs
    
    
    
 
#print("{} (AUC = {})".format(names[ii], round(roc_auc, 2)))
        
#print(dic_auc)   

f = open('dic_auc_folds.txt', 'w+')
f.write(json.dumps(dic_auc))


"""

"""
#with open(Text_file_path) as f:
#data = f.read()
#data = ''.join(data)


#opt = [ "DEEP", "MASK", "U", "SK", "ATT", "PP", "SWIN", "TRAN"] #
#names = ["Deeplab v3+", "MaskRCNN", "U-Net", "SK-U-Net", "ATT-D-U-Net", "ATT-U-Net", "UNet ++", "Swin-U-Net", "Trans-U-Net"]


#
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')    
#plt.legend(loc="lower right")
#plt.savefig("ULTRASOUND_ROC.png")

opt = [ "DEEP", "MASK", "U", "SK", "DENSE", "ATT", "PP", "SWIN", "TRAN"] #
names = ["DeepLabv3+", "Mask R-CNN", "U-Net", "Sk-U-Net", "Att-D-U-Net", "Att-U-Net", "U-Net++", "Swin-U-Net", "Trans-U-Net"]    


f = open("tprfpr/dic_tpr.txt", 'r')
f1 = json.loads(f.read())

g = open("tprfpr/dic_fpr.txt", 'r')
g1 = json.loads(g.read())
for item in range(0, len(opt)):
    tpr = f1[opt[item]]
    fpr = g1[opt[item]]
    #tpr = np.array(tpr)
    #fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
    print(auc(fpr, tpr))
    #print("{} (AUC = {})".format(names[item], round(roc_auc, 2)))
    #print("Name: {}, tpr_std: {}, fpr_std: {}").format(names[item])
    #roc_auc = auc(fpr,tpr)
    #print("{} (AUC = {})".format(names[item], round(roc_auc, 2)))
    #plt.plot(fpr, tpr, label="{} (AUC = {})".format(names[item], round(roc_auc, 2)))

#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')    
#plt.legend(loc="lower right")
#plt.savefig("ULTRASOUND_ROC.png")
"""
"""
def load_imgs(dic):
    imgs = []
    
    img_list = os.listdir(dic)
    for item in img_list:
        P2 = os.path.join(dic, item)
        img = np.load(P2)
        if img.shape == (224,224,0):

            img = np.zeros((img.shape[0], img.shape[1]))
        #print(np.min(img), np.max(img))
        imgs.append(img)
        #break
        
    imgs = np.array(imgs)
    return imgs

def load_masks(dic):
    masks = []
    img_list = os.listdir(dic)
    for item in img_list:
        #print(item)
        P2 = os.path.join(dic, item)
        mask = cv2.imread(P2, 0)
        #mask = mask//255image1 = image1/255
        mask = mask/255
        mask[mask >= 1] = 1
        mask = mask.astype(np.uint8)
        
        #mask[mask < 0.5] = 0
        #print(np.min(mask), np.max(mask))
        
        masks.append(mask)
        #break
        
    masks = np.array(masks)
    return masks

opt = [ "DEEP", "MASK", "U", "SK", "DENSE", "ATT", "PP", "SWIN", "TRAN"] #
names = ["Deeplab v3+", "MaskRCNN", "U-Net", "SK-U-Net", "ATT-D-U-Net", "ATT-U-Net", "UNet ++", "Swin-U-Net", "Trans-U-Net"]
#num_length = os.listdir()

MASKS = load_masks("ALL_IMAGES/masks")
ground_truth_labels = MASKS.ravel()
dic_tpr = {}
dic_fpr = {}
 
for ii in range(0, len(opt)):
    
    print(opt[ii])
    
    IMGS = load_imgs(opt[ii])
    #print(MASKS.shape, IMGS.shape)
    score_value = IMGS.ravel()
    fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr)
    print("{} (AUC = {})".format(names[ii], round(roc_auc, 2)))
    #MASKS = np.squeeze(MASKS)
    #cv2.imwrite("{}.png".format(ii), MASKS*255)
    #cv2.imwrite("{}.png".format(ii), IMGS*255)
    #plt.plot(fpr, tpr, label="{} (AUC = {})".format(names[ii], round(roc_auc, 2)))
    dic_tpr[opt[ii]] = list(tpr)
    dic_fpr[opt[ii]] = list(fpr)
    

f = open('dic_tpr.txt', 'w+')
f.write(json.dumps(dic_tpr))

g = open('dic_fpr.txt', 'w+')
g.write(json.dumps(dic_fpr))
"""

