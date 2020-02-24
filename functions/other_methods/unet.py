import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt

class _unet:
    def __init__(self):
        print('unet')

    def return_crop_size(self,size,levels_no):
        s = size
        for i in range(levels_no):
            s = int((s - 4) / 2)
        for i in range(levels_no):
            s = int((s - 4) * 2)
        return s


    def downsample_level(self,input,filter_size1,filter_size2,conv1_name,conv2_name,crop_size,conv2_3dim,last_layer=0,levels_no=0):
        conv1 = tf.layers.conv2d(input, filters=filter_size1, kernel_size=[3, 3], strides=(1, 1),
                                   padding='valid',
                                   activation=None, name=conv1_name, dilation_rate=(1, 1))

        conv2 = tf.layers.conv2d(conv1, filters=filter_size2, kernel_size=[3, 3], strides=(1, 1),
                                   padding='valid',
                                   activation=None, name=conv2_name, dilation_rate=(1, 1))
        crop = []
        pool=[]


        if last_layer==0:
            pool = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
            # if crop_size<=conv2_3dim:
            crop = conv2[:,
                   tf.to_int32(conv2_3dim / 2)-tf.to_int32(crop_size / 2)-1:
                   tf.to_int32(conv2_3dim / 2)+tf.to_int32(crop_size / 2),
                   tf.to_int32(conv2_3dim / 2)-tf.to_int32(crop_size / 2)-1:
                   tf.to_int32(conv2_3dim / 2) + tf.to_int32(crop_size / 2),:]

        return conv2,pool, crop

    def upsampling_level(self,input,deconv1_name,deconv2_name,filter_size1,filter_size2,filter_size3,conc,last_layer=0):
        de_conv = tf.layers.conv2d_transpose(input, filters=filter_size1, kernel_size=[3, 3], strides=(2, 2),
                                             padding='valid')
        concat = tf.concat([conc,de_conv], 3)
        conv1 = tf.layers.conv2d(concat, filters=filter_size2, kernel_size=[3, 3], strides=(1, 1),
                                 padding='valid',
                                 activation=None, name=deconv1_name, dilation_rate=(1, 1))

        conv2 = tf.layers.conv2d(conv1, filters=filter_size3, kernel_size=[3, 3], strides=(1, 1),
                                 padding='valid',
                                 activation=None, name=deconv2_name, dilation_rate=(1, 1))
        if last_layer!=0:
            conv2 = tf.layers.conv2d(conv2, filters=2, kernel_size=[1, 1], strides=(1, 1),
                                     padding='valid',
                                     activation=tf.nn.softmax, name='last_layer_deconv', dilation_rate=(1, 1))


        return conv2

    def unet(self,image,dim):
        level_number=3
        ds1_conv2_3rd_dim=dim-4
        ds2_conv2_3rd_dim =tf.to_int32(ds1_conv2_3rd_dim/2-4)
        ds3_conv2_3rd_dim =tf.to_int32(ds2_conv2_3rd_dim/2-4)
        ds4_conv2_3rd_dim =tf.to_int32(ds3_conv2_3rd_dim/2-4)

        ds3_crop=ds4_conv2_3rd_dim*2+1
        ds2_crop=(ds3_crop-4)*2+1
        ds1_crop=(ds2_crop-4)*2+1

        ds4_crop = 0

        [conv_ds1, ds1, conc1] = self.downsample_level(image, 64, 64, 'conv1_1', 'conv1_2', crop_size=ds1_crop,
                                                       conv2_3dim=ds1_conv2_3rd_dim, last_layer=0, levels_no=level_number)
        [conv_ds2, ds2, conc2] = self.downsample_level(ds1, 128, 128, 'conv2_1', 'conv2_2', crop_size=ds2_crop,
                                                       conv2_3dim=ds2_conv2_3rd_dim, last_layer=0, levels_no=level_number - 1)
        [conv_ds3, ds3, conc3] = self.downsample_level(ds2, 256, 256, 'conv3_1', 'conv3_2', crop_size=ds3_crop,
                                                       conv2_3dim=ds3_conv2_3rd_dim, last_layer=0, levels_no=level_number - 2)
        [conv_ds4, ds4, conc4] = self.downsample_level(ds3, 512, 512, 'conv4_1', 'conv4_2', crop_size=ds4_crop,
                                                       conv2_3dim=ds4_conv2_3rd_dim, last_layer=1, levels_no=level_number - 3)

        us1 = self.upsampling_level(conv_ds4, 'deconv1_1', 'deconv1_2', 256, 256, 256,conc3,last_layer=0)
        us2 = self.upsampling_level(us1, 'deconv2_1', 'deconv2_2', 128, 128, 128,conc2,last_layer=0)
        us3 = self.upsampling_level(us2, 'deconv3_1', 'deconv3_2', 64, 64, 64,conc1,last_layer=1)
        # us4 = self.upsampling_level(us3, 'deconv4_1', 'deconv4_2', 64, 64, 64,conc1,last_layer=1)

        return us3
