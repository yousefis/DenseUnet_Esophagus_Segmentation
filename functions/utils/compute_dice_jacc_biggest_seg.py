import os, time
import tensorflow as tf
import SimpleITK as sitk
#import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt
import datetime
import scipy.misc
from densenet_unet import _densenet_unet
from read_data import _read_data
from measurements import _measure


densnet_unet_config=[6,8,8,8,6]
compression_coefficient=.7
growth_rate=2
ext=''.join(map(str, densnet_unet_config))

_rd = _read_data()
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/prostate_test/'

test_CTs, test_GTVs=_rd.read_imape_path(test_path)

img_height=512
img_padded_size = 519
seg_size = 505
_meas=_measure()
jj=[]
dd=[]
in_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_14/densenet_unet_output_volume_'+ext+'/'
out_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_14/densenet_unet_output_volume_'+ext+'_Dilate/'
# data_dir = [join(out_path, f) for f in listdir(out_path) if isfile(join(out_path, f))]
labels=[]
for img_indx in range(len(test_CTs)):
    d=[]
    j=[]
    ss=str(test_CTs[img_indx]).split("/")
    nm = ss[8] + '_' + ss[9]
    path=in_path+nm+'.mha'
    GT_image = sitk.ReadImage(''.join(test_GTVs[int(img_indx)]))
    GT_image = sitk.GetArrayFromImage(GT_image)
    GT_image = (GT_image) / GT_image.mean()

    res = sitk.ReadImage(in_path+nm+'.mha')

    # Dilate
    filter = sitk.BinaryDilateImageFilter()
    filter.SetKernelRadius(3).SetForegroundValue(1)
    dilated = filter.Execute(res)
    sitk.Show(127 * (res + dilated), "Dilate by 5")

    res = sitk.GetArrayFromImage(res)
    res = (res) / res.mean()




    for i in range(res.shape[0]):
        tmp=GT_image[i][0:511,0:511]
        [dice, jaccard] = _meas.compute_dice_jaccard(tmp[int(512/2)-int(505/2)-1:int(512/2)+int(505/2),
                                                     int(512 / 2) - int(505 / 2) - 1:int(512/2)+int(505/2)],
                                                     res[i][:][:])

        d.append(dice)
        j.append(jaccard)



    _meas.plot_diagrams(d, j, nm,[], out_path)
    jj.append(np.sum(j)/len(np.where(j)[0]))
    dd.append(np.sum(d)/len(np.where(j)[0]))
    labels.append(nm)

_meas.plot_diagrams( dd, jj,'Average',labels,out_path)
