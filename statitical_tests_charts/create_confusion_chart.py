import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
"""
Labels = ["Mask R-CNN", "Trans-U-Net", "Sk-U-Net", "Swin-U-Net", "Deeplabv3+", "Att-U-Net", "U-Net", "U-Net++", "Att-D-U-Net"]

array = [[1,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],
         [0.019,1,0.900,0.900,0.024,0.001,0.001,0.001,0.001],
         [0.001,0.900,1,0.900,0.399,0.025,0.015,0.007,0.001],
         [0.001,0.604,0.900,1,0.457,0.033,0.020,0.009,0.001],
         [0.001,0.472,0.900,0.900,1,0.900,0.900,0.843,0.001],
         [0.001,0.099,0.765,0.900,0.900,1,0.900,0.900,0.001],
         [0.001,0.073,0.698,0.900,0.900,0.900,1,0.900,0.001],
         [0.001,0.052,0.625,0.900,0.900,0.900,0.900,1,0.001],
         [0.001,0.001,0.001,0.001,0.001,0.001,0.010,0.015,1]]

df_cm = pd.DataFrame(array, Labels, Labels)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1, rc={'figure.figsize':(15,12)}) # for label size
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size
ax.xaxis.tick_top()
ax.set_xlabel(`1"GFG X")
plt.savefig("testing.png")
"""

import pandas as pd
import numpy as np

