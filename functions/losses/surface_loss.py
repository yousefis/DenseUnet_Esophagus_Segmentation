import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf


def surface_loss(logits, labels,dis_map,gtv_vol):
    n_classes = 2
    y_pred = tf.reshape(logits, [-1, n_classes])
    y_true = tf.reshape(labels, [-1, n_classes])
    y_pred = tf.nn.softmax(y_pred)
    loss=(y_true*(1-y_pred)+y_pred*(1-y_true))*dis_map
    return tf.reduce_sum(loss)/gtv_vol

if __name__=='__main__':
    path= '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07052020_000/result/'
    gt_nm ='zz266545623_zz266545623_gtv.mha'#TEST108_2017-09-18_gtv.mha'#''LPRO_2013-04-26_gtv.mha'
    pm_nm ='zz266545623_zz266545623_fuzzy.mha'#TEST108_2017-09-18_fuzzy.mha'#''LPRO_2013-04-26_fuzzy.mha'
    pm_nm ='zz266545623_zz266545623_fuzzy.mha'#TEST108_2017-09-18_fuzzy.mha'#''LPRO_2013-04-26_fuzzy.mha'

    logits = sitk.GetArrayFromImage(sitk.ReadImage(path + pm_nm))
    labels = sitk.GetArrayFromImage(sitk.ReadImage(path + gt_nm))
    labels = sitk.GetArrayFromImage(sitk.ReadImage(path + gt_nm))
    gtv_vol= np.size(np.where(labels)[0])

    surface_loss(logits, labels, dis_map, gtv_vol)

