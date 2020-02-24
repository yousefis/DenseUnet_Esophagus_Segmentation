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


def penalize_mask_generator(gtv_name,ctnm):
    conn1 = (np.ones([5,5,5])>0)
    conn2 = (np.ones([5,5,5])>0)

    penalize_tag='_penalize'
    print(gtv_name)
    [scan, voxel_size, origin, direction] = rd.read_volume(gtv_name)
    scan2=scan
    input_1 = np.atleast_1d(scan2.astype(np.bool))
    E = morphology.binary_erosion(input_1, conn1)
    D = morphology.binary_dilation(input_1, conn2)
    S = np.int8((np.int8(D) - np.int8(E)) > 0)
    edited_scan = np.asarray(S)
    predicted_label = sitk.GetImageFromArray(edited_scan.astype(np.int16))
    predicted_label.SetDirection(direction=direction)
    predicted_label.SetOrigin(origin=origin)
    predicted_label.SetSpacing(spacing=voxel_size)

    name =ctnm.split('.mha')[0] + penalize_tag + '.mha'
    # print(name)

    sitk.WriteImage(predicted_label,name)

rd=_read_data(2, train_tag='train/', validation_tag='validation/', test_tag='test/',
                     img_name='CT_re113.mha', label_name='GTV_re113.mha')
CTs, GTVs, Torsos = rd.read_imape_path2('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v2/')
num_cores = multiprocessing.cpu_count()
res=Parallel(n_jobs=num_cores)(
        delayed(penalize_mask_generator)(gtv_name=GTVs[i],ctnm=CTs[i])
        for i in range(0,len(GTVs) ))#len(GTVs)