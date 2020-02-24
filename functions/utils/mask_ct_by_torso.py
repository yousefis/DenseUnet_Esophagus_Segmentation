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

def mask_ct_by_torso(torso_name,ct_name,i):

    [torso, voxel_size, origin, direction] = rd.read_volume(torso_name)
    [ct, voxel_size, origin, direction] = rd.read_volume(ct_name)


    # mask gtv by torso:
    torso_z = np.where(torso == 0)
    try:
        ct[torso_z] = -1024
    except:
        print('s')

    segmentation = np.asarray(ct)
    predicted_label = sitk.GetImageFromArray(segmentation.astype(np.short))
    predicted_label.SetDirection(direction=direction)
    predicted_label.SetOrigin(origin=origin)
    predicted_label.SetSpacing(spacing=voxel_size)
    new_ct_name=ct_name.replace('113','113z')

    sitk.WriteImage(predicted_label,new_ct_name)
    print(i,new_ct_name)

rd=_read_data(2, train_tag='train/', validation_tag='validation/', test_tag='test/',
                     img_name='CT_re113.mha', label_name='GTV_re113.mha')
CTs, GTVs, Torsos = rd.read_imape_path2('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3/')
num_cores = multiprocessing.cpu_count()-3
res=Parallel(n_jobs=num_cores)(
        delayed(mask_ct_by_torso)(torso_name=Torsos[i],ct_name=CTs[i],i=i)
        for i in range(len(GTVs) ))#len(GTVs)