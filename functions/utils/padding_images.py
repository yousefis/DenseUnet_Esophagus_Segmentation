import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

from functions.read_data2 import _read_data
from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
#for padding images with the size of patches!

def image_padding( img, padLowerBound, padUpperBound, constant):
    filt = sitk.ConstantPadImageFilter()
    padded_img = filt.Execute(img,
                              padLowerBound,
                              padUpperBound,
                              constant)
    return padded_img
def padding_images(ct_name,torso_name,gt_name,i):
    CT_image1 = sitk.ReadImage(''.join(ct_name))
    voxel_size = CT_image1.GetSpacing()
    origin = CT_image1.GetOrigin()
    direction = CT_image1.GetDirection()

    GTV_image1 = sitk.ReadImage(''.join(gt_name))
    Torso_image1 = sitk.ReadImage(''.join(torso_name))

    bth = sitk.BinaryThresholdImageFilter()
    AA = (np.unique(sitk.GetArrayFromImage(GTV_image1)) == [0, 255])
    XX = np.bitwise_and(AA[0], AA[1])
    if XX:
        GTV_image1 = bth.Execute(GTV_image1, 255, 255, 1, 0)
    if len(np.unique(sitk.GetArrayFromImage(GTV_image1)))!=2:
        print('eeeeeeeeeeeeeeeeeeeerrrrrrrrrrrrrrrrrrrroooooooooooooooooooooooooooooooooooooooooooooooooooooor %s'%gt_name)
    patch_window=87
    padd_zero = patch_window * 2 + 2
    # with TicTocGenerator('padding'):
    CT_image1 = image_padding(img=CT_image1,
                                   padLowerBound=[int(padd_zero / 2) + 1, int(padd_zero / 2) + 1,
                                                  int(padd_zero / 2) + 1],
                                   padUpperBound=[int(padd_zero / 2), int(padd_zero / 2), int(padd_zero / 2)],
                                   constant=-1024)

    GTV_image1 = image_padding(img=GTV_image1,
                                    padLowerBound=[int(padd_zero / 2) + 1, int(padd_zero / 2) + 1,
                                                   int(padd_zero / 2) + 1],
                                    padUpperBound=[int(padd_zero / 2), int(padd_zero / 2), int(padd_zero / 2)],
                                    constant=0)

    Torso_image1 = image_padding(img=Torso_image1,
                                      padLowerBound=[int(padd_zero / 2) + 1, int(padd_zero / 2) + 1,
                                                     int(padd_zero / 2) + 1],
                                      padUpperBound=[int(padd_zero / 2), int(padd_zero / 2), int(padd_zero / 2)],
                                      constant=0)



    new_ct_name=ct_name.replace('re113','re113_pad'+str(patch_window))
    new_gt_name=gt_name.replace('re113','re113_pad'+str(patch_window))
    new_torso_name=torso_name.replace('re113','re113_pad'+str(patch_window))
    if (CT_image1.GetSpacing()!=(1,1,3)):
        print(i,'error')
        exit(1)
    if (GTV_image1.GetSpacing()!=(1,1,3)):
        print(i,'error')
        exit(1)
    if (Torso_image1.GetSpacing()!=(1,1,3)):
        print(i,'error')
        exit(1)

    ct=sitk.GetImageFromArray(sitk.GetArrayFromImage(CT_image1).astype(np.short))
    gtv=sitk.GetImageFromArray(sitk.GetArrayFromImage(GTV_image1).astype(np.int8))
    torso=sitk.GetImageFromArray(sitk.GetArrayFromImage(Torso_image1).astype(np.int8))
    ct.SetDirection(direction)
    ct.SetSpacing(voxel_size)
    ct.SetOrigin(origin)

    gtv.SetDirection(direction)
    gtv.SetSpacing(voxel_size)
    gtv.SetOrigin(origin)

    torso.SetDirection(direction)
    torso.SetSpacing(voxel_size)
    torso.SetOrigin(origin)

    sitk.WriteImage(ct,new_ct_name)
    sitk.WriteImage(gtv,new_gt_name)
    sitk.WriteImage(torso,new_torso_name)
    print(i,new_ct_name)


_rd = _read_data(data=2,train_tag='train/', validation_tag='validation/', test_tag='test/',
                 img_name='CT_re113.mha', label_name='GTV_re113.mha',torso_tag='Torso_re113.mha')


train_CTs, train_GTVs, train_Torso, train_penalize, \
        validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
        test_CTs, test_GTVs, test_Torso, test_penalize=_rd.read_data_path(fold=0)

# CTs, GTVs, Torsos = _rd.read_image_path3('/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3')
# CTs1, GTVs1, Torsos1 = _rd.read_image_path3('/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v4')
num_cores = multiprocessing.cpu_count()
res=Parallel(n_jobs=num_cores)(
        delayed(padding_images)(ct_name=train_CTs[i],torso_name=train_Torso[i],gt_name=train_GTVs[i],i=i)
        for i in range(400, len(train_GTVs) ))

res=Parallel(n_jobs=num_cores)(
        delayed(padding_images)(ct_name=validation_CTs[i],torso_name=validation_Torso[i],gt_name=validation_GTVs[i],i=i)
        for i in range(len(train_GTVs) ))

res=Parallel(n_jobs=num_cores)(
        delayed(padding_images)(ct_name=test_CTs[i],torso_name=test_Torso[i],gt_name=test_GTVs[i],i=i)
        for i in range(len(train_GTVs) ))