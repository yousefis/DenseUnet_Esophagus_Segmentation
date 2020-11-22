
from datetime import datetime
from functions.Segmentation.densenet_seg_surface_attention_spatial_skip_ch_attention import dense_seg
import numpy as np
import tensorflow as tf
import os

if __name__=='__main__':
    np.random.seed(1)
    tf.set_random_seed(1)
    server_path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/' #path to the home dir
    Logs= 'Log_2019_09_23/Dataset3/' #path to log dir
    fold=0 #split number
    mixed_precision=True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    GPU= tf.test.is_gpu_available(
        cuda_only=False,
        min_cuda_compute_capability=None
    )
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S") #make a name for logging dir


    dc12=dense_seg(data=2,
                   densnet_unet_config=[3,3,5,3,3],
                   compression_coefficient=.75,
                   sample_no=2000000,
                   validation_samples=1980,
                   no_sample_per_each_itr=1000,
                   train_tag='',
                   validation_tag='',
                   test_tag='',
                   img_name='',label_name='', torso_tag='',
                   log_tag='-train1-'+date_time+str(fold),
                   tumor_percent=0.75, # percentage of patches including the tumors
                   Logs=Logs,
                   fold=fold,
                   server_path=server_path,
                   growth_rate=4)
    dc12.run_net()


