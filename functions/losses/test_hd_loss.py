import functions.losses.weighted_hd as weighted_hd
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-04172020_140/result/'

gtv='LPRO_2013-05-14_4DCT_0_gtv.mha'
out='LPRO_2013-05-14_4DCT_0_result.mha'

Gtv= sitk.GetArrayFromImage(sitk.ReadImage(path+gtv))
Out= sitk.GetArrayFromImage(sitk.ReadImage(path+out))
res= weighted_hd.Weighted_Hausdorff_loss(np.expand_dims(Gtv[117,:,:],0),np.expand_dims(Out[117,:,:],0))
sess= tf.Session()
sess.run(res)
print(res)