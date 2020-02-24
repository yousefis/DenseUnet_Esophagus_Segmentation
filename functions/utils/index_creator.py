import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

from read_data import _read_data
from joblib import Parallel, delayed
import multiprocessing

import pandas as pd

def penalize_mask_generator(gtv_name,indx):

    nm = gtv_name.split('.mha')[0] + '.xlsx'
    print(indx,nm)
    [scan, voxel_size, origin, direction] = rd.read_volume(gtv_name)
    where=np.where(scan)

    df = pd.DataFrame({'x': where[0],
                       'y': where[1],
                       'z': where[2]})
    writer = pd.ExcelWriter(nm)
    df.to_excel(writer,'Sheet1')
    writer.save()

rd=_read_data(2, train_tag='train/', validation_tag='validation/', test_tag='test/',
                     img_name='CT_re113.mha', label_name='GTV_re113.mha')
CTs, GTVs, Torsos = rd.read_imape_path2('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3/')
num_cores = multiprocessing.cpu_count()
res=Parallel(n_jobs=num_cores)(
        delayed(penalize_mask_generator)(gtv_name=GTVs[i],indx=i)
        for i in range(len(GTVs) ))#len(GTVs)