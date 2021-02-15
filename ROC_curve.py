import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndi 
from sklearn.metrics import auc

"""
ROC CURVE using sklearn auc
Input: npy files containing ROC information. 
layout: ROC/TPR/FILE.npy

To get ROC data use, code below
model = unet()   
model.load_weights(PATH)

PREDICTED_MASK = model.predict(X_TEST)
ground_truth_labels = Y_TEST_test.ravel() # we want to make them into vectors
score_value = PREDICTED_MASK.ravel() # we want to make them into vectors
fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
roc_auc = auc(fpr,tpr)
np.save("ROC/TPR_SKUNET.npy", tpr)
np.save("ROC/FPR_SKUNET.npy", fpr)

"""

L_TPR_DEEPLAB = np.load("ROC\TPR\TPR_DEEPLAB.npy") # load TPR data
L_FPR_DEEPLAB = np.load("ROC\FPR\FPR_DEEPLAB.npy") # load FPR data
#L_TPR_DEEPLAB = [log1p(x) for x in L_TPR_DEEPLAB]
#L_FPR_DEEPLAB = [log1p(x) for x in L_FPR_DEEPLAB] 
AUC_DEEPLAB = auc(L_FPR_DEEPLAB, L_TPR_DEEPLAB) # calculate AUC

L_TPR_MASKRCNN = np.load("ROC\TPR\TPR_MASKRCNN.npy")
L_FPR_MASKRCNN = np.load("ROC\FPR\FPR_MASKRCNN.npy")
#L_TPR_MASKRCNN = [log1p(x) for x in L_TPR_MASKRCNN]
#L_FPR_MASKRCNN = [log1p(x) for x in L_FPR_MASKRCNN] 
AUC_MASKRCNN = auc(L_FPR_MASKRCNN,L_TPR_MASKRCNN)

L_TPR_UNET = np.load("ROC\TPR\TPR_UNET.npy")
L_FPR_UNET = np.load("ROC\FPR\FPR_UNET.npy")
#L_TPR_UNET = [log1p(x) for x in L_TPR_UNET]
#L_FPR_UNET = [log1p(x) for x in L_FPR_UNET] 
AUC_UNET = auc(L_FPR_UNET,L_TPR_UNET)

L_TPR_SKUNET = np.load("ROC\TPR\TPR_SKUNET.npy")
L_FPR_SKUNET = np.load("ROC\FPR\FPR_SKUNET.npy")
#L_TPR_SKUNET = [log1p(x) for x in L_TPR_SKUNET]
#L_FPR_SKUNET = [log1p(x) for x in L_FPR_SKUNET] 
AUC_SKUNET = auc(L_FPR_SKUNET,L_TPR_SKUNET) 

L_TPR_DENSEUNET = np.load("ROC\TPR\TPR_DENSEUNET.npy")
L_FPR_DENSEUNET = np.load("ROC\FPR\FPR_DENSEUNET.npy")
#L_TPR_DENSEUNET = [log1p(x) for x in L_TPR_DENSEUNET]
#L_FPR_DENSEUNET = [log1p(x) for x in L_FPR_DENSEUNET] 
AUC_DENSEUNET = auc(L_FPR_DENSEUNET,L_TPR_DENSEUNET) 

L_TPR_ATTUNET = np.load("ROC\TPR\TPR_ATTUNET.npy")
L_FPR_ATTUNET = np.load("ROC\FPR\FPR_ATTUNET.npy")
#L_TPR_ATTUNET = [log1p(x) for x in L_TPR_ATTUNET]
#L_FPR_ATTUNET = [log1p(x) for x in L_FPR_ATTUNET] 
AUC_ATTUNET = auc(L_FPR_ATTUNET,L_TPR_ATTUNET)

L_TPR_UNETPP = np.load("ROC\TPR\TPR_UNETPP.npy")
L_FPR_UNETPP = np.load("ROC\FPR\FPR_UNETPP.npy")
#L_TPR_UNETPP = [log1p(x) for x in L_TPR_UNETPP]
#L_FPR_UNETPP = [log1p(x) for x in L_FPR_UNETPP]  
AUC_UNETPP = auc(L_FPR_UNETPP,L_TPR_UNETPP) 


plt.plot(L_FPR_DEEPLAB, L_TPR_DEEPLAB, label="Deeplab v3+ (area = %0.2f)" % AUC_DEEPLAB)
plt.plot(L_FPR_MASKRCNN, L_TPR_MASKRCNN, label="MaskRCNN (area = %0.2f)" % AUC_MASKRCNN)
plt.plot(L_FPR_UNET, L_TPR_UNET, label="U-Net (area = %0.2f)" % AUC_UNET)
plt.plot(L_FPR_SKUNET, L_TPR_SKUNET, label="SK-U-Net (area = %0.2f)" % AUC_SKUNET)
plt.plot(L_FPR_DENSEUNET, L_TPR_DENSEUNET, label="ATT-D-U-Net (area = %0.2f)" % AUC_DENSEUNET)
plt.plot(L_FPR_ATTUNET, L_TPR_ATTUNET, label="ATT-U-Net (area = %0.2f)" % AUC_ATTUNET)
plt.plot(L_FPR_UNETPP, L_TPR_UNETPP, label="UNet ++ (area = %0.2f)" % AUC_UNETPP)

plt.xscale("log")#remove for not log scale
plt.title("ROC CURVE")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([10**-3.5, 1])
#plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")


