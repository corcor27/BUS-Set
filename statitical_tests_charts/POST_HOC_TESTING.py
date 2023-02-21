import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import pandas as pd
import matplotlib.image as mpimg 
import scipy.ndimage as ndi 

import pandas as pd
import scipy.stats as stats
import pingouin as pg
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.stats.multitest
"""
NAMES =[]
pvals = []
EXCEL_PATH = r"D:\deeplabv3\PREDICTION_FOLDS_DICE_SCORES.xlsx"
data = pd.read_excel(EXCEL_PATH)
cols = data.columns[1:9]
for col in cols:
    tukey = pairwise_tukeyhsd(endog=data[col],groups=data['FOLD'],alpha=0.05)
    #print(tukey)
    df1 = data[data.FOLD  == 1]
    df2 = data[data.FOLD  == 2]
    df3 = data[data.FOLD  == 3]
    df4 = data[data.FOLD  == 4]
    df5 = data[data.FOLD  == 5]
    one_anova = f_oneway(df1[col], df2[col], df3[col], df4[col], df5[col])
    print(col, one_anova[1])
    NAMES.append(col)
    pvals.append(one_anova[1])

decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(pvals=pvals, alpha=0.05, method='holm')

print(f'Original p-values: \t {pvals}')
print(f'Adjusted p-values: \t {adj_pvals}')

  

values = []
models = []

EXCEL_PATH = r"D:\deeplabv3\PREDICTION_FOLDS_ACC_SCORES.xlsx"
data = pd.read_excel(EXCEL_PATH)
cols = data.columns[1:10]
for col in cols:
    for row in range(0, data.shape[0]):
        values.append(data[col].iloc[row])
        models.append(col)
dframe = pd.DataFrame()
dframe["mod"] = models
dframe["values"] = values
print(dframe)
#tukety = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=dframe['values'], groups=dframe['mod'], alpha=0.01)
#print(tukety)

#cpt = pg.pairwise_tukey(dv=col, between="FOLD", effsize='cohen', data=data)
    #ttest = pg.pairwise_ttests(dv=col, between='FOLD', data=data).round(3)
    #print(cpt)
"""
EXCEL_PATH = r"D:\deeplabv3\Man_shape.xlsx"
dframe = pd.read_excel(EXCEL_PATH)
print(dframe)
#dframe = pd.DataFrame()
#dframe["mod"] = models
#dframe["values"] = values
#print(dframe)
#print(dframe)
tukety = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=dframe['values'], groups=dframe['mod'], alpha=0.01)
print(tukety)

#cpt = pg.pairwise_tukey(dv=col, between="FOLD", effsize='cohen', data=data)
    #ttest = pg.pairwise_ttests(dv=col, between='FOLD', data=data).round(3)
    #print(cpt)
