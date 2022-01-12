
from datetime import datetime
from functions.Segmentation.densenet_seg_surface_attention_spatial_skip_ch_attention import dense_seg
import numpy as np
import tensorflow as tf
import os
from argparse import ArgumentParser
# server_path= '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'
# Logs='Log_2019_09_23/Dataset3/'
def parse_inputs():
    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, required=False, default=0, help="Fold number for k-fold cross validation ")
    parser.add_argument("--server_path", type=str, required=True, help="path to the home dir on the server")
    parser.add_argument("--log", type=str, required=True, help="path to log dir")
    parser.add_argument("--compression_coefficient", type=float, required=False, default=.75, help="Compression coefficient")
    parser.add_argument("--sample_no", type=int, required=False, default=500, help="Sample no")
    parser.add_argument("--validation_samples", type=int, required=False, default=1980, help="Validation samples")
    parser.add_argument("--no_sample_per_each_itr", type=int, required=False, default=1000, help="No sample per each itr")
    parser.add_argument("--growth_rate", type=int, required=False, default=4, help="Growth rate")
    parser.add_argument("--tumor_percent", type=float, required=False, default=0.75, help="Percentage of patches including the tumors")
    parser.add_argument("--learning_decay", type=float, required=False, default=0.95, help="Learning decay")
    parser.add_argument("--learning_rate", type=float, required=False, default=1E-5, help="Learning rate")
    parser.add_argument("--beta_rate", type=float, required=False, default=0.05, help="Beta rate")
    parser.add_argument("--img_padded_size", type=int, required=False, default=519, help="Size of padding")
    parser.add_argument("--seg_size", type=int, required=False, default=505, help="Size of segmentation")
    parser.add_argument("--GTV_patchs_size", type=int, required=False, default=49, help="Size of GTV patches")
    parser.add_argument("--patch_window", type=int, required=False, default=63, help="Patch window")
    parser.add_argument("--batch_no", type=int, required=False, default=7, help="Batch size of training")
    parser.add_argument("--batch_no_validation", type=int, required=False, default=30, help="Batch size of validation")
    parser.add_argument("--display_step", type=int, required=False, default=100, help="Display step of training")
    parser.add_argument("--display_validation_step", type=int, required=False, default=1, help="Display step of validation")
    parser.add_argument("--total_epochs", type=int, required=False, default=10, help="Epoch no")
    parser.add_argument("--dropout_keep", type=float, required=False, default=.5, help="Dropout keep")
    parser.add_argument("--img_size", type=int, required=False, default=.5, help="Image size")
    parser.add_argument("--data_type", type=int, required=False, default=2, help="The value is 1 if the data only includes 3D scans and 2 if the data includes 4D scans")
    parser.add_argument("--densnet_unet_config", nargs="+", default=[3,3,5,3,3], help="The config of the network, make sure it doesn't return zero no of layers after downsampling")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    np.random.seed(1)
    tf.set_random_seed(1)
    args = parse_inputs()
    server_path = args.server_path
    Logs = args.log
    fold = args.fold
    mixed_precision = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    GPU= tf.test.is_gpu_available(
        cuda_only=False,
        min_cuda_compute_capability=None
    )
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S") #make a name for logging dir

    dc12 = dense_seg(data=args.data_type,
                    densnet_unet_config = args.densnet_unet_config,
                    compression_coefficient = args.compression_coefficient,
                    sample_no=args.sample_no,
                    validation_samples=args.validation_samples,
                    no_sample_per_each_itr=args.no_sample_per_each_itr,
                    log_tag='-train1-'+date_time+str(fold),
                    tumor_percent=args.tumor_percent,
                    Logs=Logs,
                    fold=fold,
                    server_path=server_path,
                    growth_rate=args.growth_rate,
                    learning_decay = args.learning_decay,
                    learning_rate = args.learning_rate,
                    beta_rate= args.beta_rate,
                    img_padded_size=args.img_padded_size,
                    seg_size=args.seg_size,
                    GTV_patchs_size=args.GTV_patchs_size,
                    patch_window= args.patch_window,
                    batch_no=args.batch_no,
                    batch_no_validation=args.batch_no_validation,
                    display_step= args.display_step,
                    display_validation_step=args.display_validation_step,
                    total_epochs = args.total_epochs,
                    dropout_keep= args.dropout_keep,
                    img_size = args.img_size)
    dc12.run_net()


