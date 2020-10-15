#call largest_component.py before this script!
import sys
import pandas as pd

sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("../..") # Adds higher directory to python modules path.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, time
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.misc
from functions.loss_func import _loss_func

import xlsxwriter
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle

# import datetime
# from densenet_unet import _densenet_unet
# from read_data import _read_data
# from measurements import _measure
# from image_class import image_class
vali=0
if vali==0:
    out_dir = 'result_vali/'
else:
    out_dir = 'result/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-NoRand-tumor25-004/'

test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_1stdataset/13331_0.75_4-cross-noRand-train1-104/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_1stdataset/13331_0.75_4-cross-noRand-train1-105/'
test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07142020_020/' #Spatial only

test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07102020_140/' #spatial and channel

# test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07052020_000/' #channel
test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-08052020_140/' #surface
test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-04172020_140/'
# test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-08242020_1950240/'
# test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-05082020_090/'

import multiprocessing
from joblib import Parallel, delayed
eps=10e-10
def read_names( tag):

    gtv_names = [join(test_path+out_dir, g) for g in [f for f in listdir(test_path + out_dir) if ~isfile(join(test_path, f)) ]\
                  if g.endswith(tag)]
    gtv_names=np.sort(gtv_names)
    return gtv_names

def read_fuzzy_gt_names():
    fuzzy=read_names(tag='_fuzzy.mha')
    gt=read_names(tag='_gtv.mha')
    return fuzzy,gt
def read_image(path_fuzzy,path_gt):
    fuzzy=sitk.GetArrayFromImage(sitk.ReadImage(path_fuzzy))
    gt=sitk.GetArrayFromImage(sitk.ReadImage(path_gt))
    return fuzzy,gt

def read_imgs(path_fuzzy, path_gt):
    [logits, labels] = read_image(path_fuzzy, path_gt)
    return logits, labels

def tp_tn_fp_fn(logits, labels,threshold,path_fuzzy,cntr):

    # y_pred = np.asarray(logits).astype(np.bool):
    y_pred = logits>=threshold
    y_true = np.asarray(labels).astype(np.bool)
    im1 = y_pred.flatten()
    im2 = y_true.flatten()
    TP_t = len(np.where((im1 == True) & (im2 == True))[0])
    TN_t = len(np.where((im1 == False) & (im2 == False))[0])
    FP_t = len(np.where((im1 == True) & (im2 == False))[0])
    FN_t = len(np.where((im1 == False) & (im2 == True))[0])

    # TP_b = len(np.where((im1 == False) & (im2 == False))[0])
    # TN_b = len(np.where((im1 == True) & (im2 == True))[0])
    # FP_b = len(np.where((im1 == False) & (im2 == True))[0])
    # FN_b = len(np.where((im1 == True) & (im2 == False))[0])

    # TP=np.array((TP_t,TP_b))
    #
    # TN = np.array((TN_t,TN_b))
    #
    # FP = np.array((FP_t,FP_b))
    #
    # FN = np.array((FN_t,FN_b))

    print('cntr:%d, threshold: %f, tp: %d, tn: %d,fp:%d,fn:%d'%(cntr,threshold,TP_t,TN_t,FP_t,FN_t))
    nm=str.split(path_fuzzy,'/')[-1]
    return nm,threshold,TP_t,TN_t,FP_t,FN_t
def Precision(TP,TN,FP,FN):
    precision=TP/(TP+FP+ eps)
    return precision

def Recall(TP,TN,FP,FN):
    recall=TP/(TP+FN+ eps)
    return recall

[path_fuzzy,path_gt]=read_fuzzy_gt_names()
step=1/200
threshold_vec=np.arange(0,1.0+2*step,step)
# threshold_vec=np.arange(0,.3,0.1)
num_cores = 10#multiprocessing.cpu_count()

# volumes=Parallel(n_jobs=num_cores)(
#         delayed(read_image)(path_fuzzy=path_fuzzy[i],
#                             path_gt=path_gt[i],i=i)
#         for i in range(len(path_fuzzy)))#
res_all=[]
for cntr in range(len(path_fuzzy)):
    xsl_nm = test_path + out_dir + str.split(str.split(path_gt[cntr], '/')[-1], '_gtv.mha')[0] + '.xlsx'
    [logits, labels] =read_imgs(path_fuzzy[cntr], path_gt[cntr])
    res=Parallel(n_jobs=num_cores)(
            delayed(tp_tn_fp_fn)(logits=logits, labels=labels,threshold=threshold_vec[i],path_fuzzy=path_fuzzy[cntr],cntr=cntr)
            for i in range(len(threshold_vec))
            )


    df = pd.DataFrame(res,
                     columns=pd.Index(['name','threshold',
                                        'TP','TN','FP','FN'],
                     name='Genus'))
    # Create a Pandas Excel writer using XlsxWriter as the engine.

    writer = pd.ExcelWriter(xsl_nm,
                            engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    # os.remove(path_fuzzy[cntr])


# TP=np.zeros(len(threshold_vec))
# TN=np.zeros(len(threshold_vec))
# FP=np.zeros(len(threshold_vec))
# FN=np.zeros(len(threshold_vec))
# for i in range(len(threshold_vec)):
#     for j in range(len(path_fuzzy)):
#         TP[i]=TP[i]+res[len(path_fuzzy)*i+j][0]
#         TN[i]=TN[i]+res[len(path_fuzzy)*i+j][1]
#         FP[i]=FP[i]+res[len(path_fuzzy)*i+j][2]
#         FN[i]=FN[i]+res[len(path_fuzzy)*i+j][3]
#
# Precision_vec=np.zeros(len(threshold_vec))
# Recall_vec=np.zeros(len(threshold_vec))
#
# for i in range(len(threshold_vec)):
#     Precision_vec[i]=Precision(TP[i],TN[i],FP[i],FN[i])
#     Recall_vec[i]=Recall(TP[i],TN[i],FP[i],FN[i])
#
# plt.plot(Precision_vec,Recall_vec)
# plt.xlabel('Precision')
# plt.ylabel('Recall')
# plt.savefig(test_path+'PR_curve')
# print('tp')