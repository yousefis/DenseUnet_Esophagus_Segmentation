#call largest_component.py before this script!
import sys
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
from scipy.ndimage import morphology

import datetime
# from read_data import _read_data
vali=0
if vali==1:
    out_dir = 'result_vali/'
else:
    out_dir = 'result/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_2nddataset/13331_0.75_4-cross-noRand-train2test2--8/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_1stdataset/13331_0.75_4-cross-noRand-train1-107/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-NoRand-tumor25-004/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_2nddataset/13331_0.75_4-cross-noRand-train2test2--6/'
# test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_1stdataset/13331_0.75_4-cross-noRand-train1-104/'
eps=10e-6
def surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds
def read_names( tag='_result.mha'):

    gtv_names = [join(test_path+out_dir, g) for g in [f for f in listdir(test_path + out_dir) if ~isfile(join(test_path, f)) \
                 and  not join(test_path, f).endswith('_fuzzy_result.mha')] if g.endswith(tag)]
    gtv_names=np.sort(gtv_names)
    return gtv_names
def tp_tn_fp_fn(logits, labels):
    y_pred = np.asarray(logits).astype(np.bool)
    y_true = np.asarray(labels).astype(np.bool)
    im1 = y_pred.flatten()
    im2 = y_true.flatten()
    TP_t = len(np.where((im1 == True) & (im2 == True))[0])
    TN_t = len(np.where((im1 == False) & (im2 == False))[0])
    FP_t = len(np.where((im1 == True) & (im2 == False))[0])
    FN_t = len(np.where((im1 == False) & (im2 == True))[0])

    TP_b = len(np.where((im1 == False) & (im2 == False))[0])
    TN_b = len(np.where((im1 == True) & (im2 == True))[0])
    FP_b = len(np.where((im1 == False) & (im2 == True))[0])
    FN_b = len(np.where((im1 == True) & (im2 == False))[0])

    TP=np.array((TP_t,TP_b))

    TN = np.array((TN_t,TN_b))

    FP = np.array((FP_t,FP_b))

    FN = np.array((FN_t,FN_b))

    return TP,TN,FP,FN
def Precision(TP,TN,FP,FN):
    precision=TP/(TP+FP+ eps)
    return precision

def Recall(TP,TN,FP,FN):
    recall=TP/(TP+FN+ eps)
    return recall
def f1_measure(TP,TN,FP,FN):
    precision=Precision(TP,TN,FP,FN)
    recall=Recall(TP,TN,FP,FN)
    f1 = 2 * (precision * recall) / (precision + recall + eps)  # f0:background, f1: tumor
    # print(f1)
    return f1

result=read_names()
result=np.sort(result)
result_lc=read_names('_result_lc.mha')
result_lc=np.sort(result_lc)
Gtv=read_names('_gtv.mha')
Gtv=np.sort(Gtv)
dsc_res=[]
dsc_res_lc=[]
dsc=[]
name_list=[]
def get_largest_component(predicted_image_path,out_image_name) :
    # read the input image
    maskImg = sitk.ReadImage(predicted_image_path)
    maskImg = sitk.Cast(maskImg, sitk.sitkUInt8)

    # initialize the connected component filter
    ccFilter = sitk.ConnectedComponentImageFilter()
    # apply the filter to the input image
    labelImg = ccFilter.Execute(maskImg)
    # get the number of labels (connected components)
    numberOfLabels = ccFilter.GetObjectCount()
    # print('numberOfLabels: %d'%numberOfLabels)
    # extract the data array from the itk object
    labelArray = sitk.GetArrayFromImage(labelImg)
    # count the voxels belong to different components
    labelSizes = np.bincount(labelArray.flatten())
    # get the largest connected component

    if len(labelSizes[1:])!=0:
        largestLabel = np.argmax(labelSizes[1:]) + 1
        # convert the data array to itk object
        outImg = sitk.GetImageFromArray((labelArray == largestLabel).astype(np.uint8))
        # output image should have same metadata as input mask image
        outImg.CopyInformation(maskImg)
        voxel_size = maskImg.GetSpacing()
        origin = maskImg.GetOrigin()
        direction = maskImg.GetDirection()
        outImg.SetDirection(direction=direction)
        outImg.SetOrigin(origin=origin)
        outImg.SetSpacing(spacing=voxel_size)
    else:
        outImg=maskImg

    # write the image to the disk
    sitk.WriteImage(outImg, out_image_name)

def overlapped_component(predicted_image_path,overlapped_name) :
    # read the input image
    maskImg = sitk.ReadImage(predicted_image_path)
    outImg =maskImg
    maskImg = sitk.Cast(maskImg, sitk.sitkUInt8)
    gtvImg = sitk.GetArrayFromImage(sitk.ReadImage(predicted_image_path.split('_result.mha')[0]+'_gtv.mha'))
    # torsoImg = sitk.GetArrayFromImage(sitk.ReadImage(predicted_image_path.split('_result.mha')[0]+'_T.mha'))
    # initialize the connected component filter
    ccFilter = sitk.ConnectedComponentImageFilter()
    # apply the filter to the input image
    labelImg = ccFilter.Execute(maskImg)
    # get the number of labels (connected components)
    numberOfLabels = ccFilter.GetObjectCount()
    # print('numberOfLabels: %d'%numberOfLabels)
    if numberOfLabels>1:
        # extract the data array from the itk object
        labelArray = sitk.GetArrayFromImage(labelImg)
        # count the voxels belong to different components
        labelSizes = np.bincount(labelArray.flatten())
        # get the largest connected component

        if len(labelSizes[1:])!=0:
            indx = np.where((labelArray * gtvImg))
            if len(indx[0]):
                largestLabel = np.unique(labelArray[indx])[0]
                # overlayLabel=
                # convert the data array to itk object
                outImg = sitk.GetImageFromArray((labelArray == largestLabel).astype(np.uint8))
                # output image should have same metadata as input mask image
                outImg.CopyInformation(maskImg)
                voxel_size = maskImg.GetSpacing()
                origin = maskImg.GetOrigin()
                direction = maskImg.GetDirection()
                outImg.SetDirection(direction=direction)
                outImg.SetOrigin(origin=origin)
                outImg.SetSpacing(spacing=voxel_size)
        else:
            outImg=maskImg

    # write the image to the disk
    sitk.WriteImage(outImg, overlapped_name)
    return  sitk.GetArrayFromImage(outImg)
def compute_tp_tn_fp_fn(result,result_lc,Gtv,lc_name,overlapped_name):
# for i in range(len(result)):#
#     get_largest_component(result,lc_name)
    res = sitk.GetArrayFromImage(sitk.ReadImage(result))
    res_lc = sitk.GetArrayFromImage(sitk.ReadImage(result_lc))
    gtv = sitk.GetArrayFromImage(sitk.ReadImage(Gtv))

    overlapped=overlapped_component(result,overlapped_name)


    if len(np.where(gtv)[0]) == 0 or len(np.where(overlapped)[0]) == 0:
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        top_prep_dis=max(np.where(gtv)[0])-max(np.where(overlapped)[0])
        bottom_prep_dis=min(np.where(gtv)[0])-min(np.where(overlapped)[0])



        [TP, TN, FP, FN] = tp_tn_fp_fn(res, gtv)
        f1 = f1_measure(TP, TN, FP, FN)
        # dsc_res.append(f1[0])

        [TP, TN, FP, FN] = tp_tn_fp_fn(res_lc, gtv)
        f1_lc = f1_measure(TP, TN, FP, FN)


        top_prep_dis_lc=max(np.where(gtv)[0])-max(np.where(overlapped)[0])
        bottom_prep_dis_lc=min(np.where(gtv)[0])-min(np.where(overlapped)[0])

        srfd = surfd(res_lc, gtv, [3, 1, 1])
        msd = srfd.mean()
        hd = srfd.max()
        rms = np.sqrt((srfd ** 2).mean())

        # dsc_res_lc.append(f1_lc[0])
        x=np.array([TP[0],TP[1],
                    TN[0],
                    TN[1], FP[0],
                    FP[1], FN[0],
                    FN[1],f1[0],f1_lc[0],
                   top_prep_dis,bottom_prep_dis,
                    top_prep_dis_lc,bottom_prep_dis_lc,
                    msd,hd,rms])
    # if not len(dsc):
    #     dsc=x
    # else:
    #     dsc=np.vstack((dsc,x))
    # name_list.append(result[i].split('/')[-1].split('_result.mha')[0])


        # print('hd:%f,msd:%f' % (hd, msd))
        print('%s: f1:%f,f1:%f ' % (result.split('/')[-1], f1[0], f1_lc[0]))
    return result.split('/')[-1].split('_result.mha')[0],x


import multiprocessing
from joblib import Parallel, delayed

num_cores =  multiprocessing.cpu_count()
res=Parallel(n_jobs=num_cores)(
        delayed(compute_tp_tn_fp_fn)(result=result[i],
                                     result_lc=result_lc[i],
                                     Gtv=Gtv[i],
                                     lc_name=result[i].split('_result.mha')[0]+'_result_lc.mha',
                                     overlapped_name=result[i].split('_result.mha')[0]+'_result_ov.mha'
                                     )
        for i in range(len(result)))#

dsc=[]
name_list=[]
tp_list=[]
tn_list=[]
fp_list=[]
fn_list=[]
for i in range(len(res)):
    if np.sum(res[i][1])==0:
        continue
    name_list.append(res[i][0])
    if not len(dsc):
        dsc=res[i][1]
    else:
        dsc=np.vstack((dsc,res[i][1]))


# Create a Pandas dataframe from some data.

df = pd.DataFrame(dsc,
                 index=name_list,
                 columns=pd.Index(['TP_tumor','TP_back',
                                    'TN_tumor','TN_back',
                                   'FP_tumor', 'FP_back',
                                   'FN_tumor', 'FN_back',
                                   'DCS-LC', 'DCS+LC',
                                   'top_prep_dis','bottom_prep_dis',
                                   'top_prep_dis_lc', 'bottom_prep_dis_lc',
                                   'msd','hd','rms'],
                 name='Genus')).round(2)


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(test_path+out_dir+'/all_dice.xlsx',
                        engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

df.plot(kind='bar',figsize=(10,4))

ax = plt.gca()
pos = []
for bar in ax.patches:
    pos.append(bar.get_x()+bar.get_width()/2.)


ax.set_xticks(pos,minor=True)
lab = []
for i in range(len(pos)):
    l = df.columns.values[i//len(df.index.values)]
    lab.append(l)

ax.set_xticklabels(lab,minor=True)
ax.tick_params(axis='x', which='major', pad=15, size=0)
plt.setp(ax.get_xticklabels(), rotation=0)

plt.margins(.05)
plt.ylim([0,1])
plt.grid()

plt.savefig(test_path+out_dir+'dsc_bar2.png')

# plt.figure()
# ax = plt.gca()
fig, ax1 = plt.subplots(figsize=(10, 6))
bp = plt.boxplot(dsc, notch=0, sym='+', vert=1, whis=1.5, showmeans=True)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
# bp=plt.boxplot(dsc, 0, 'rs')
ax.set_xticklabels(['-CS', '+CS'])
plt.ylim([0,1])
plt.grid()
plt.savefig(test_path+out_dir+'dsc_bp2.png')
# plt.show()