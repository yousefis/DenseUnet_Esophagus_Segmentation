from functions.densenet_classify_miccai import denseUnet_classify_miccai
import numpy as np
import tensorflow as tf
#Training only on the second dataset
fold=2
np.random.seed(1)
tf.set_random_seed(1)



dc12=denseUnet_classify_miccai( data=2,
                     densnet_unet_config=[3,4,4,4,3],
                     compression_coefficient=.5,
                     growth_rate=4,
                     sample_no=2000000,
                     validation_samples=1980,
                     no_sample_per_each_itr=1000,
                     train_tag='', validation_tag='', test_tag='',
                     img_name='',label_name='', torso_tag='',
                     log_tag='-cross-noRand-train1-'+str(fold),min_range=-1000,max_range=3000,
                     tumor_percent=.75,
                     other_percent=.25,
                     Logs='Log_2018-08-15/00-miccai_densenet/',
                     fold=fold)
#6 th was the first
dc12.run_net()


