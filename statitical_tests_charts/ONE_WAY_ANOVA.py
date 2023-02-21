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

FOLD_MEANS = []
for ii in range(1, 6):
    nam = []
    means = []
    EXCEL_PATH = r"D:\deeplabv3\FOLD_{}_RESULTS.xlsx".format(ii)
    data = pd.read_excel(EXCEL_PATH)
    cols = data.columns[:8]
    for col in cols:
        nam.append(col)
        vals = list(data[col])
        mean = np.mean(vals)
        means.append(mean)
    means = np.array(means)
    means = np.expand_dims(means, axis=1)
    reshape = np.reshape(means, (means.shape[1], means.shape[0]))
    FOLD_MEANS.append(means.reshape(1,means.shape[0]))
    
FOLD_MEANS = np.squeeze(np.array(FOLD_MEANS))

print(FOLD_MEANS.shape)

frame = pd.DataFrame()

testing_folds = ["FOLD1", "FOLD2", "FOLD3", "FOLD4", "FOLD5"]

frame["Dataset"] = testing_folds

for name in range(0, len(nam)):
    mod = nam[name]
    frame[mod] = FOLD_MEANS[:, name]
#frame.to_excel("statistical_table.xlsx")

#print(frame.head())
    
print(frame)

nova = pg.rm_anova(frame.iloc[:,1:])
print(nova)
#pt = pg.pairwise_tukey(dv='weight', between='group', data=frame)
#print(pt)
#fvalue, pvalue = stats.f_oneway(FOLD_MEANS[])
#print(fvalue, pvalue)
