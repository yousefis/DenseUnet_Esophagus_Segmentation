import os, time
import sys
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.misc
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("../..") # Adds higher directory to python modules path.


# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
vali=0
import datetime

test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_2nddataset/13331_0.75_4-cross-noRand-train2test2--8/']
test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_1stdataset/13331_0.75_4-cross-noRand-train1-106/']
test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_1stdataset/13331_0.75_4-cross-noRand-train1-107/']
test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/00_Seperate_training_2nddataset/13331_0.75_4-cross-noRand-train2test2--6/']

def read_names( test_path,out_dir):

    gtv_names = [join(test_path+out_dir, g) for g in [f for f in listdir(test_path + out_dir) if ~isfile(join(test_path, f)) \
                 and  not join(test_path, f).endswith('_fuzzy.mha')] if g.endswith('_result.mha')]

    return gtv_names

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
    print('numberOfLabels: %d'%numberOfLabels)
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
def get_largest_component2(predicted_image_path,out_image_name) :
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
    print('numberOfLabels: %d'%numberOfLabels)
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
    # else:



    # write the image to the disk
    sitk.WriteImage(outImg, out_image_name)

# test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/5foldcrossvalidation/13331_0.75_4-cross-noRand-00110/']
# test_path.append('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/5foldcrossvalidation/13331_0.75_4-cross-noRand-2-00110/')
# test_path.append('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-noRand-2-001/')
# test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-NoRand-tumor25-000/']
# test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-noRand-2-002/']
# test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-noRand-2-001/']
# test_path=['/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/5foldcrossvalidation/13331_0.75_4-cross-noRand-00113/']
import multiprocessing
from joblib import Parallel, delayed
if vali==1:
    out_dir = 'result_vali/'
else:
    out_dir = 'result/'
for i in range(len(test_path)):
    hi = 1
    gtv_names=read_names(test_path[i],out_dir)
    gtv_names.sort()
    # for i in range(len(gtv_names)):
    #     print(gtv_names[i])
    #     result_lc_name=
    #     get_largest_component2(gtv_names[i],result_lc_name)


    num_cores = multiprocessing.cpu_count()
    res=Parallel(n_jobs=num_cores)(
            delayed(get_largest_component2)(gtv_names[i],gtv_names[i].split('_result.mha')[0]+'_result_lc.mha'
                                         )
            for i in range(len(gtv_names)))#len(result)