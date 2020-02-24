import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt


# !!

class _dense_net:
    def __init__(self, compression_coefficient=.5, class_no=2, growth_rate=2):
        print('create object _dense_net')
        self.compres_coef = compression_coefficient
        self.class_no = class_no
        self.growth_rate = growth_rate
        self.kernel_size1 = 1
        self.kernel_size2 = 3

    def transition_layer(self, dense_out1, transition_name, conv_name, kernel_size=[1, 1],
                         padding='same', activation=None, dilation_rate=(1, 1), pool_size=[2, 2], strides=2):
        with tf.name_scope(transition_name):
            filter = int(dense_out1.get_shape()[3].value * self.compres_coef)
            conv1 = tf.layers.conv2d(inputs=dense_out1, filters=filter, kernel_size=kernel_size, padding=padding,
                                     activation=activation,
                                     name=conv_name, dilation_rate=dilation_rate)
            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=strides)
            return conv1

    # ========================
    def dense_block(self, input, filter, padding='same', activation=None, name='dense_sub_block', flag=0,
                    concat_flag=0):
        with tf.name_scope(name):
            db_conv1 = tf.layers.conv2d(input, filters=filter * 4, kernel_size=self.kernel_size1, padding=padding,
                                        activation=activation)
            db_conv2 = tf.layers.conv2d(db_conv1, filters=filter, kernel_size=self.kernel_size2, padding=padding,
                                        activation=activation)
            db_concat = tf.concat([input, db_conv2], 3)
        return db_concat



    # ========================
    def dense_loop(self, loop, input, crop_size, db_size, padding='same', activation=None, name='dense_block', flag=0,
                   concat_flag=0):
        with tf.name_scope(name):
            output = input
            for i in range(loop):
                output = self.dense_block(output, filter=self.growth_rate,
                                          padding=padding, activation=activation, name='dense_sub_block' + str(i),
                                          flag=flag, concat_flag=concat_flag)

        cropped = output[:,
                  tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                  tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2),
                  tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                  tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2), :]
        #
        # cropped = output[:,
        #           np.int32(db_size / 2) - np.int32(crop_size / 2) - 1:
        #           np.int32(db_size / 2) + np.int32(crop_size / 2),
        #           np.int32(db_size / 2) - np.int32(crop_size / 2) - 1:
        #           np.int32(db_size / 2) + np.int32(crop_size / 2), :]
        return output, cropped

    # ========================

    def dens_net(self, image, is_training, dropout_rate, dim=101):

        # dim2=101
        # db_size1 = np.int32(dim2 / 2)
        # db_size2 = np.int32(db_size1 / 2)
        # crop_size1 = np.int32((np.int32(db_size2 /2-2)*2 + 1.0))
        # crop_size2 = np.int32((crop_size1 - 2) * 2 + 1)
        # db_size0=0
        # crop_size0=0

        db_size1 = tf.to_int32(dim / 2)
        db_size2 = tf.to_int32(db_size1 / 2)
        crop_size1 = tf.add(tf.multiply(tf.add(tf.div(db_size2, 2), -2), 2), 1)
        crop_size2 = tf.add(tf.multiply(tf.add(crop_size1, -2), 2), 1)

        db_size0 = tf.to_int32(0)
        crop_size0 = tf.to_int32(0)

        # pre_conv = tf.layers.conv2d(image, filters=4 * self.growth_rate, kernel_size=[3, 3], strides=(1, 1), padding='same',
        #                             activation=None, name='pre_conv', dilation_rate=(1, 1))
        # pre_bn=tf.layers.batch_normalization(pre_conv)
        # with tf.name_scope('preprocessing'):
        #     pre_reu=tf.nn.relu(pre_conv)
        # pre_pool = tf.layers.max_pooling2d(pre_reu, pool_size=[2, 2], strides=2)
        # dense block #1

        [dense_out1, conctmp] = self.dense_loop(loop=6, input=image, crop_size=crop_size0, db_size=db_size0,
                                                padding='same', activation=None, name='dense_block_6', concat_flag=1)

        pool1 = self.transition_layer(dense_out1, 'transition_1', conv_name='conv1', kernel_size=[1, 1], padding='same',
                                      activation=None,
                                      dilation_rate=(1, 1), pool_size=[2, 2], strides=2)
        # ========================
        [dense_out2, conc1] = self.dense_loop(loop=8, input=pool1, crop_size=crop_size2, db_size=db_size1,
                                              padding='same', activation=None, name='dense_block_8')

        pool2 = self.transition_layer(dense_out2, 'transition_2', conv_name='conv2', kernel_size=[1, 1], padding='same',
                                      activation=None,
                                      dilation_rate=(1, 1), pool_size=[2, 2], strides=2)
        # =========================================================
        [dense_out3, conc2] = self.dense_loop(loop=10, input=pool2, crop_size=crop_size1, db_size=db_size2,
                                              padding='same', activation=None, name='dense_block_10')

        pool3 = self.transition_layer(dense_out3, 'transition_3', conv_name='conv3', kernel_size=[1, 1], padding='same',
                                      activation=None,
                                      dilation_rate=(1, 1), pool_size=[2, 2], strides=2)
        # =========================================================
        [dense_out4, conc3] = self.dense_loop(loop=10, input=pool2, crop_size=crop_size1, db_size=db_size2,
                                              padding='same', activation=None, name='dense_block_10')

        # pool4 = self.transition_layer(dense_out4, 'transition_3', conv_name='conv3', kernel_size=[1, 1], padding='same',
        #                               activation=None,
        #                               dilation_rate=(1, 1), pool_size=[2, 2], strides=2)


        # post_bn = tf.layers._normalization(dense_out4)
        post_reu=tf.nn.relu(dense_out4)

        # classification layer:
        with tf.name_scope('classification_layer'):
            # post_rm = tf.reduce_mean(pool3, [1, 2], name='global_avg_pool', keep_dims=True)
            fc1 = tf.layers.conv2d(pool3, filters=64, kernel_size=[3,3], padding='same', strides=(1, 1),
                                 activation=tf.nn.relu, dilation_rate=(1, 1),name='fc1')

            dropout = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=is_training,name='droup_out')

            y = tf.layers.conv2d(dropout, filters=self.class_no, kernel_size=[3,3], padding='same', strides=(1, 1),
                                 activation=tf.nn.softmax, dilation_rate=(1, 1), name='fc2')

        print(' total numbe of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        #
        # res = tf.layers.conv2d(pro_conv2, filters=2, kernel_size=[1, 1], strides=(1, 1),
        #                        padding='valid',
        #                        activation=tf.nn.softmax, name='last_layer_deconv', dilation_rate=(1, 1))





        '''h_fc1 = tf.contrib.layers.fully_connected(post_rm,20,activation_fn=tf.nn.relu)

        #Fully connected layer #2
        #h_fc2 = tf.layers.conv2d(h_fc1,512,[3,3],padding="valid", strides=(1,1),activation=tf.nn.relu, dilation_rate=(1, 1))
        h_fc2 = tf.contrib.layers.fully_connected(h_fc1,8,activation_fn=tf.nn.relu)

        #Fully connected layer #3
        #y =  tf.layers.conv2d(h_fc2,2,[3,3],padding="valid", strides=(1,1),activation=tf.nn.relu, dilation_rate=(1, 1))
        y = tf.contrib.layers.fully_connected(h_fc2,class_no,activation_fn=tf.nn.relu)
'''
        return y

    # ========================
    # def dens_net01(self, image, is_training, dropout_rate):
    #         pre_conv = tf.layers.conv2d(image, filters=2 * self.growth_rate, kernel_size=[7, 7], strides=(2, 2),
    #                                     padding='same',
    #                                     activation=None, name='pre_conv', dilation_rate=(1, 1))
    #         # pre_bn=tf.layers.batch_normalization(pre_conv)
    #         with tf.name_scope('preprocessing'):
    #             pre_reu = tf.nn.relu(pre_conv)
    #             pre_pool = tf.layers.max_pooling2d(pre_reu, pool_size=[2, 2], strides=2)
    #         # dense block #1
    #
    #         dense_out1 = self.dense_loop(loop=6, input=pre_pool,
    #                                      padding='same', activation=None, name='dense_block_6')
    #
    #         pool1 = self.transition_layer(dense_out1, 'transition_1', conv_name='conv1', kernel_size=[1, 1],
    #                                       padding='same', activation=None,
    #                                       dilation_rate=(1, 1), pool_size=[2, 2], strides=2)
    #         # ========================
    #         dense_out2 = self.dense_loop(loop=8, input=pool1,
    #                                      padding='same', activation=None, name='dense_block_8')
    #
    #         pool2 = self.transition_layer(dense_out2, 'transition_2', conv_name='conv2', kernel_size=[1, 1],
    #                                       padding='same',
    #                                       activation=None,
    #                                       dilation_rate=(1, 1), pool_size=[2, 2], strides=2)
    #         # =========================================================
    #         dense_out3 = self.dense_loop(loop=10, input=pool2,
    #                                      padding='same', activation=None, name='dense_block_10')
    #
    #         pool3 = self.transition_layer(dense_out3, 'transition_3', conv_name='conv3', kernel_size=[1, 1],
    #                                       padding='same',
    #                                       activation=None,
    #                                       dilation_rate=(1, 1), pool_size=[2, 2], strides=2)
    #
    #         # pro_conv=tf.layers.conv2d(inputs=pool3, filters=100, kernel_size=[2, 2], padding='valid', activation=None,
    #         #                  name='f22f', dilation_rate=(1, 1))
    #         # =========================================================
    #         # dense_out4 = self.dense_loop(loop=12, input=pool3,
    #         #                              padding='same', activation=None,name='dense_block_12')
    #         # pool4 = tf.layers.max_pooling2d(inputs=dense_out4, pool_size=[2, 2], strides=2)
    #
    #
    #         de_conv1 = tf.layers.conv2d_transpose(pool3, filters=1024, kernel_size=[1, 1], strides=(2, 2),
    #                                               padding='valid')
    #         dense_out4 = self.dense_loop(loop=10, input=de_conv1,
    #                                      padding='same', activation=None, name='dense_block_10_2')
    #         pro_conv1 = tf.layers.conv2d(dense_out4, filters=dense_out4.shape[3], kernel_size=3, padding='valid',
    #                                      activation=None)
    #
    #         # =========================================================
    #
    #         de_conv2 = tf.layers.conv2d_transpose(pro_conv1, filters=800, kernel_size=[1, 1], strides=(2, 2),
    #                                               padding='valid')
    #         dense_out5 = self.dense_loop(loop=8, input=de_conv2,
    #                                      padding='same', activation=None, name='dense_block_8_2')
    #         pro_conv2 = tf.layers.conv2d(dense_out5, filters=dense_out5.shape[3], kernel_size=2, padding='valid',
    #                                      activation=None)
    #
    #         # =========================================================
    #
    #         de_conv3 = tf.layers.conv2d_transpose(pro_conv2, filters=560, kernel_size=[1, 1], strides=(2, 2),
    #                                               padding='valid')
    #         dense_out6 = self.dense_loop(loop=6, input=de_conv3,
    #                                      padding='same', activation=None, name='dense_block_6_2')
    #
    #         pro_conv3 = tf.layers.conv2d(dense_out6, filters=dense_out6.shape[3], kernel_size=2, padding='valid',
    #                                      activation=None)
    #         # =========================================================
    #         de_conv4 = tf.layers.conv2d_transpose(pro_conv3, filters=360, kernel_size=[1, 1], strides=(2, 2),
    #                                               padding='valid')
    #         dense_out7 = self.dense_loop(loop=2, input=de_conv4,
    #                                      padding='same', activation=None, name='dense_block_2')
    #
    #         pro_conv4 = tf.layers.conv2d(dense_out7, filters=dense_out7.shape[3], kernel_size=2, padding='valid',
    #                                      activation=None)
    #         # =========================================================
    #         de_conv5 = tf.layers.conv2d_transpose(pro_conv4, filters=260, kernel_size=[1, 1], strides=(2, 2),
    #                                               padding='valid')
    #         dense_out8 = self.dense_loop(loop=2, input=de_conv5,
    #                                      padding='same', activation=None, name='dense_block_2')
    #
    #         pro_conv5 = tf.layers.conv2d(dense_out8, filters=dense_out8.shape[3], kernel_size=2, padding='valid',
    #                                      activation=None)
    #
    #         # post_bn = tf.layers._normalization(dense_out4)
    #         post_reu = tf.nn.relu(pro_conv5)
    #
    #         # classification layer:
    #         with tf.name_scope('classification_layer'):
    #             # post_rm = tf.reduce_mean(post_reu, [1, 2], name='global_avg_pool', keep_dims=True)
    #             fc1 = tf.layers.conv2d(post_reu, filters=64, kernel_size=[1, 1], padding='valid', strides=(1, 1),
    #                                    activation=tf.nn.relu, dilation_rate=(1, 1), name='fc1')
    #
    #             dropout = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=is_training, name='droup_out')
    #
    #             y = tf.layers.conv2d(dropout, filters=self.class_no, kernel_size=[1, 1], padding='valid',
    #                                  strides=(1, 1),
    #                                  activation=None, dilation_rate=(1, 1), name='fc2')
    #
    #         '''h_fc1 = tf.contrib.layers.fully_connected(post_rm,20,activation_fn=tf.nn.relu)
    #
    #         #Fully connected layer #2
    #         #h_fc2 = tf.layers.conv2d(h_fc1,512,[3,3],padding="valid", strides=(1,1),activation=tf.nn.relu, dilation_rate=(1, 1))
    #         h_fc2 = tf.contrib.layers.fully_connected(h_fc1,8,activation_fn=tf.nn.relu)
    #
    #         #Fully connected layer #3
    #         #y =  tf.layers.conv2d(h_fc2,2,[3,3],padding="valid", strides=(1,1),activation=tf.nn.relu, dilation_rate=(1, 1))
    #         y = tf.contrib.layers.fully_connected(h_fc2,class_no,activation_fn=tf.nn.relu)    '''
    #     return y
    # ========================



    def vgg(self, image):
        conv1 = tf.layers.conv2d(image, filters=64, kernel_size=[3, 3], strides=(2, 2), padding='valid',
                                 activation=None, name='conv1', dilation_rate=(1, 1))

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3], strides=(2, 2), padding='valid',
                                 activation=None, name='conv2', dilation_rate=(1, 1))

        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3, 3], strides=(2, 2), padding='valid',
                                 activation=None, name='conv3', dilation_rate=(1, 1))

        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

        y = tf.layers.conv2d(pool3, filters=2, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                             activation=None, dilation_rate=(1, 1))
        return y
        # =======================
