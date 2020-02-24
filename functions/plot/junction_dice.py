from os import listdir

import SimpleITK as sitk
import matplotlib.pyplot as plt
# import math as math
import numpy as np
from os.path import isfile, join

log_tag='13331_0.75_4-cross-NoRand-tumor25-004/'
Log = 'Log_2018-08-15/cross_validation/'
out_dir = 'result/'
test_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+log_tag

plot_tag='junction_dice'
eps=1E-5

jj = []
dd = []
dice_boxplot0 = []
dice_boxplot1= []
dice_boxplot = []

jacc_boxplot = []
jacc_boxplot0 = []
jacc_boxplot1 = []

f1_boxplot0_esophgus=[]
f1_boxplot1_esophgus=[]
f1_boxplot_av_esophgus=[]

f1_boxplot0_stomach=[]
f1_boxplot1_stomach=[]
f1_boxplot_av_stomach=[]

fpr_av=[]
fnr_av=[]
xtickNames = []
name_list=[]


fpr0=[]
fpr1=[]

fnr0=[]
fnr1=[]

sp0=[]
sp1=[]

recall0=[]
recall1=[]
recall_av=[]
presicion0=[]
presicion1=[]
presicion_av=[]
def read_names( ):

    gtv_names = [join(test_path+out_dir, f) for f in listdir(test_path+out_dir) if ~isfile(join(test_path, f)) and join(test_path, f).endswith('_gtv.mha')]
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
def dice( im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    print('intersect:%d, sum1:%d, sum2:%d'%(intersection.sum(),im1.sum() , im2.sum()))
    d=2. * intersection.sum() / (im1.sum() + im2.sum() + eps)

    return d
def f1_measure(TP,TN,FP,FN):
    precision=Precision(TP,TN,FP,FN)
    recall=Recall(TP,TN,FP,FN)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # f0:background, f1: tumor
    print(f1)
    return f1

def jaccard( im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Jaccard coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / (im1.sum() + im2.sum() - intersection.sum() + eps)

def Sp(TP,TN,FP,FN):

    Sp=TN/(TN+FP+eps)
    return Sp
def FPR(TP,TN,FP,FN):

    fpr=FP/(TN+FP+eps)
    return fpr

def FNR(TP,TN,FP,FN):
    fnr=FN/(TP+FN+eps)
    return fnr

def Precision(TP,TN,FP,FN):
    precision=TP/(TP+FP+ eps)
    return precision

def Recall(TP,TN,FP,FN):
    recall=TP/(TP+FN+ eps)
    return recall
def junction_tp_tn_fp_fn(gtv_name,result_name,junction_slice_no):
    gtv=sitk.ReadImage( gtv_name)
    _zz_img_gt=sitk.GetArrayFromImage(gtv)
    result=sitk.ReadImage( result_name)
    _zz_img = sitk.GetArrayFromImage(result)
    [TP_esophgus,TN_esophgus,FP_esophgus,FN_esophgus]=tp_tn_fp_fn(_zz_img[0:junction_slice_no,:,:], _zz_img_gt[0:junction_slice_no,:,:])
    [TP_stomach,TN_stomach,FP_stomach,FN_stomach]=tp_tn_fp_fn(_zz_img[junction_slice_no:-1,:,:], _zz_img_gt[junction_slice_no:-1,:,:])
    return TP_esophgus,TN_esophgus,FP_esophgus,FN_esophgus,TP_stomach,TN_stomach,FP_stomach,FN_stomach


lc=''

gtv_names=read_names()
gtv_names.sort()
for gtv_name in gtv_names:
    name_list.append(gtv_name.split(out_dir)[1].split('.mha')[0].split('_gtv')[0].split('_CT')[0])
    name_tmp=gtv_name.split(out_dir)[1].split('.mha')[0].split('_gtv')[0].split('_CT')[0]
    if len(name_tmp.split('TEST'))==2:
        continue
    elif len(name_tmp.split('LPRO'))==2:
        junction_slice_no=123#140#51
    elif len(name_tmp.split('LPSC'))==2:
        junction_slice_no=140#128 #40
    elif len(name_tmp.split('LWWI'))==2:
        junction_slice_no=115#131#42
    elif len(name_tmp.split('RAWO'))==2:
        junction_slice_no=132#183#141
    elif len(name_tmp.split('RGLA'))==2:#does not contain stomach
        junction_slice_no=-1#-
    elif len(name_tmp.split('RJSC'))==2:
        junction_slice_no=127#147#87


    result_name=gtv_name.split('_gtv.mha')[0]+'_result'+lc+'.mha'
    [TP_esophgus,TN_esophgus,FP_esophgus,FN_esophgus,
     TP_stomach,TN_stomach,FP_stomach,FN_stomach]=junction_tp_tn_fp_fn(gtv_name, result_name,junction_slice_no)

    f1_esophgus=f1_measure(TP_esophgus,TN_esophgus,FP_esophgus,FN_esophgus)
    print('name: %s => f1:%f,f1:%f'%(gtv_name,f1_esophgus[0],f1_esophgus[1]))
    f1_boxplot0_esophgus.append(f1_esophgus[0])
    f1_boxplot1_esophgus.append(f1_esophgus[1])
    f1_boxplot_av_esophgus.append((f1_esophgus[0]+f1_esophgus[1])/2)

    f1_stomach = f1_measure(TP_stomach,TN_stomach,FP_stomach,FN_stomach)
    print('name: %s => f1:%f,f1:%f' % (gtv_name, f1_stomach[0], f1_stomach[1]))
    f1_boxplot0_stomach.append(f1_stomach[0])
    f1_boxplot1_stomach.append(f1_stomach[1])
    f1_boxplot_av_stomach.append((f1_stomach[0] + f1_stomach[1]) / 2)


# plt.close('all')
f1_bp0 = []
f1_bp1 = []
f1_bp_av = []
f1_bp0.append((f1_boxplot0_esophgus))
f1_bp1.append((f1_boxplot1_esophgus))
f1_bp_av.append((f1_boxplot_av_esophgus))
plt.figure()
plt.boxplot(f1_bp0)
plt.title('Tumor Dice value for all the images esophagus'+plot_tag)
plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+log_tag+'/'+out_dir+'f1_bp_tumor_esophagus'+lc+'.png')


f1_bp0 = []
f1_bp1 = []
f1_bp_av = []
f1_bp0.append((f1_boxplot0_stomach))
f1_bp1.append((f1_boxplot1_stomach))
f1_bp_av.append((f1_boxplot_av_stomach))
plt.figure()
plt.boxplot(f1_bp0)
plt.title('Tumor Dice value for all the images stomach'+plot_tag)
plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+log_tag+'/'+out_dir+'f1_bp_tumor_stomach'+lc+'.png')


