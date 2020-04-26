import pandas as pd
from os import listdir

import SimpleITK as sitk
import matplotlib.pyplot as plt
# import math as math
import numpy as np
from os.path import isfile, join

log_tag = '23432_0.75_4-train1-03222020_180/'
Log = '/Log_2019_09_23/Dataset3/'
test_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+log_tag
out_dir = '/result/'
plot_tag='1_tumor_dice'
eps=1E-5

jj = []
dd = []
dice_boxplot0 = []
dice_boxplot1= []
dice_boxplot = []

jacc_boxplot = []
jacc_boxplot0 = []
jacc_boxplot1 = []

f1_boxplot0=[]
f1_boxplot1=[]
f1_boxplot_av=[]

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
def plot_diagrams(gtv_name,result_name):
    gtv=sitk.ReadImage( gtv_name)
    _zz_img_gt=sitk.GetArrayFromImage(gtv)
    result=sitk.ReadImage( result_name)
    _zz_img = sitk.GetArrayFromImage(result)
    [TP,TN,FP,FN]=tp_tn_fp_fn(_zz_img, _zz_img_gt)
    return TP,TN,FP,FN


lc=''

gtv_names=read_names()
gtv_names.sort()
for gtv_name in gtv_names:
    name_list.append(gtv_name.split('result/')[1].split('.mha')[0].split('_gtv')[0].split('_CT')[0])
    result_name=gtv_name.split('_gtv.mha')[0]+'_result'+lc+'.mha'
    [TP, TN, FP, FN]=plot_diagrams(gtv_name, result_name)

    f1=f1_measure(TP,TN,FP,FN)
    print('name: %s => f1:%f,f1:%f'%(gtv_name,f1[0],f1[1]))
    f1_boxplot0.append(f1[0])
    f1_boxplot1.append(f1[1])
    f1_boxplot_av.append((f1[0]+f1[1])/2)

    fpr=FPR(TP,TN,FP,FN)
    fpr0.append(fpr[0])
    fpr1.append(fpr[1])
    fpr_av.append((fpr[0]+fpr[1])/2)

    fnr = FNR(TP,TN,FP,FN)
    fnr0.append(fnr[0])
    fnr1.append(fnr[1])
    fnr_av.append((fnr[0]+fnr[1])/2)

    precision = Precision(TP,TN,FP,FN)
    presicion0.append(precision[0])
    presicion1.append(precision[1])
    presicion_av.append((precision[0]+precision[1])/2)

    recall = Recall(TP,TN,FP,FN)
    recall0.append(recall[0])
    recall1.append(recall[1])
    recall_av.append((recall[0]+recall[1])/2)




plt.close('all')
f1_bp0 = []
f1_bp1 = []
f1_bp_av = []
f1_bp0.append((f1_boxplot0))
f1_bp1.append((f1_boxplot1))
f1_bp_av.append((f1_boxplot_av))
plt.figure()
plt.boxplot(f1_bp0)
plt.title('Tumor Dice value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'f1_bp_tumor'+lc+'.png')



items = {'patient': name_list,
        'dice': f1_bp0[0]
        }

df = pd.DataFrame(items, columns = ['patient', 'dice'])


df.to_excel(test_path+'/'+out_dir+"dice.xlsx")

plt.figure()
plt.boxplot(f1_bp1)
plt.title('Background Dice value for all the images '+plot_tag)
plt.savefig(test_path+'/'+out_dir+'f1_bp_background'+lc+'.png')

plt.figure()
plt.boxplot(f1_bp_av)
plt.title('Average Dice value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'f1_bp_average'+lc+'.png')
#----------------------
fpr_bp0 = []
fpr_bp0.append((fpr0))
plt.figure()
plt.boxplot(fpr_bp0, 0, '')
plt.title('FPR Tumor value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'fpr_bp_tumor'+lc+'.png')

fpr_bp1 = []
fpr_bp1.append((fpr1))
plt.figure()
plt.boxplot(fpr_bp1)
plt.title('FPR Background value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'fpr_bp_background'+lc+'.png')


fpr_bp = []
fpr_bp.append((fpr_av))
plt.figure()
plt.boxplot(fpr_bp)
plt.title('FPR Average value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'fpr_bp_average'+lc+'.png')

#----------------------
fnr_bp0 = []
fnr_bp0.append((fnr0))
plt.figure()
plt.boxplot(fnr_bp0)
plt.title('FNR Tumor value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'fnr_bp_tumor'+lc+'.png')

fnr_bp1 = []
fnr_bp1.append((fnr1))
plt.figure()
plt.boxplot(fnr_bp1)
plt.title('FNR Background value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'fnr_bp_background'+lc+'.png')


fnr_bp = []
fnr_bp.append((fnr_av))
plt.figure()
plt.boxplot(fnr_bp)
plt.title('FNR Average value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'fnr_bp_average'+lc+'.png')
#----------------------
pres_bp0 = []
pres_bp0.append((presicion0))
plt.figure()
plt.boxplot(pres_bp0)
plt.title('Precision value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'precision_bp_tumor'+lc+'.png')

pres_bp1 = []
pres_bp1.append((presicion1))
plt.figure()
plt.boxplot(pres_bp1)
plt.title('Precision Background value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'precision_bp_background'+lc+'.png')


pres_bp = []
pres_bp.append((presicion_av))
plt.figure()
plt.boxplot(pres_bp)
plt.title('Precision Average value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'precision_bp_average'+lc+'.png')
#----------------------
recall_bp0 = []
recall_bp0.append((recall0))
plt.figure()
plt.boxplot(recall_bp0)
plt.title('Recall value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'recall_bp_tumor'+lc+'.png')

recall_bp1 = []
recall_bp1.append((recall1))
plt.figure()
plt.boxplot(recall_bp1)
plt.title('Recall Background value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'recall_bp_background'+lc+'.png')


recall_bp = []
recall_bp.append((recall_av))
plt.figure()
plt.boxplot(recall_bp)
plt.title('Recall Average value for all the images'+plot_tag)
plt.savefig(test_path+'/'+out_dir+'recall_bp_average'+lc+'.png')
#----------------------
plt.figure()
d_bp = []
d_bp.append((f1_boxplot0))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, f1_boxplot0, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.45)
plt.title('Dice all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'dice_bar'+lc+'.png')

#----------------------
plt.figure()

fnr_bar0 = []
fnr_bar0.append((fnr0))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, fnr0, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('FNR Background all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'fnr_background_bar'+lc+'.png')


#----------------------
plt.figure()

fnr_bar1 = []
fnr_bar1.append((fnr1))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, fnr1, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('FNR Tumor all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'fnr_tumor_bar'+lc+'.png')


#----------------------
plt.figure()

fpr_bar0 = []
fpr_bar0.append((fpr0))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, fpr0, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('FPR Background all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'fpr_background_bar'+lc+'.png')


#----------------------
plt.figure()

fpr_bar1 = []
fpr_bar1.append((fpr1))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, fpr1, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('FPR tumor all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'fpr_tumor_bar'+lc+'.png')


#----------------------
plt.figure()

recall_bar0 = []
recall_bar0.append((recall0))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, recall0, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('Recall Background all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'recall_background_bar'+lc+'.png')


#----------------------
plt.figure()

recall_bar = []
recall_bar.append((recall1))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, recall1, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('Recall tumor all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'recall_tumor_bar'+lc+'.png')

#----------------------
plt.figure()

recall_bar = []
recall_bar.append((recall_av))
xs = [i for i,_ in enumerate(name_list)]

plt.bar(xs, recall_av, align='center')
plt.xticks(xs, name_list, rotation='vertical')
plt.margins(.05)
plt.subplots_adjust(bottom=0.25)
plt.title('Recall Average all images'+plot_tag)
plt.grid()
plt.savefig(test_path+'/'+out_dir+'recall_average_bar'+lc+'.png')
