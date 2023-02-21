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
from statsmodels.multivariate.manova import MANOVA


#FILE = r"D:\deeplabv3\MORE_MANOVA.xlsx"
FILE = r"D:\deeplabv3\Man_shape.xlsx"
data = pd.read_excel(FILE)

#manova_result = MANOVA.from_formula('ACC + IOU + DICE ~ Model', data)
manova_result = MANOVA.from_formula('ELON + CIR + DWR ~ Model', data)
print(manova_result.mv_test())

