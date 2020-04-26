import matplotlib.pyplot as plt
from functions.loss_func import _loss_func
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
if __name__=='__main__':
    loss_instance=_loss_func()
    penalize1 = sitk.GetArrayFromImage(sitk.ReadImage('/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3/LAHO/2014-02-03/distancemap_CT_re113_pad87.mha'))
    y1 = sitk.GetArrayFromImage(sitk.ReadImage(
        '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3/LAHO/2014-02-03/CT_Torso_re113_pad87.mha'))
    label1 = sitk.GetArrayFromImage(sitk.ReadImage(
        '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3/LAHO/2014-02-03/GTV_CT_re113_pad87.mha'))
    torso=y1
    label1=np.eye(2)[label1]
    y1=np.eye(2)[y1]

    label1=label1[np.newaxis,...]#1
    y1=y1[np.newaxis,...]#1
    penalize1=penalize1[np.newaxis,...,np.newaxis]#0
    # y=np.random.randint(1,10,(30,283,677,677,1))
    # y1=np.repeat(y1, 30, axis=0)
    # label1=np.repeat(label1, 30, axis=0)
    # penalize1=np.repeat(penalize1, 30, axis=0)

    y = tf.placeholder(tf.float32, shape=[None, None, None, None, 2])
    label = tf.placeholder(tf.float32, shape=[None, None, None, None, 2])
    penalize = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])

    [mm,multiply,sqr,penalized_loss,
     soft_dice_coef, logt, lbl] = loss_instance.dice_plus_distance_penalize(logits=y, labels=label, penalize=penalize,torso=torso)
    sess=tf.Session()
    [mm1,multiply1,sqr1,penalized_loss1,soft_dice_coef1] = sess.run([mm,multiply,sqr,penalized_loss, soft_dice_coef],
                                               feed_dict={y:y1,
                                                          label:label1,
                                                          penalize:penalize1})
    print(penalized_loss1)
    print(soft_dice_coef1)
    # plt.imshow(penalized_loss1[0, 93, :, :, 0])
    # plt.show()
    print(3)
