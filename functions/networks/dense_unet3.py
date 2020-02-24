import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import time


# !!

class _densenet_unet:
    def __init__(self, densnet_unet_config,compression_coefficient, growth_rate, class_no=2):
        print('create object _densenet_unet')
        self.compres_coef = compression_coefficient
        self.class_no = class_no
        self.growth_rate = growth_rate
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.config=densnet_unet_config
        self.log_ext = '_'+''.join(map(str, self.config)) + '_' + str(
            self.compres_coef) + '_' + str(self.growth_rate)
        self.seed=200

    def transition_layer(self,
                         dense_out1,
                         transition_name,
                         conv_name,
                         is_training_bn,
                         conv_pool_name,
                         db_size,crop_size,
                         kernel_size=[1, 1, 1],
                         padding='same',
                         activation=None,
                         dilation_rate=(1, 1,1),
                         pool_size=[2, 2, 2],
                         strides=(2, 2, 2),
                         bn_flag = False):
        with tf.name_scope(transition_name):
            filter = int(dense_out1.get_shape()[4].value * self.compres_coef)
            if bn_flag==False:
                conv1 = tf.layers.conv3d(inputs=dense_out1, filters=filter, kernel_size=kernel_size, padding=padding,
                                         activation=activation,
                                        name=conv_name + self.log_ext, dilation_rate=dilation_rate)
            else:
                conv1 = tf.layers.conv3d(inputs=dense_out1, filters=filter, kernel_size=kernel_size, padding=padding,
                                         activation=None,
                                         name=conv_name + self.log_ext, dilation_rate=dilation_rate)
                bn1 = tf.layers.batch_normalization(conv1, training=is_training_bn,renorm=False)
                bn1 = tf.nn.leaky_relu(bn1)
                conv1=bn1

            cropped = conv1[:,
                      tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2), :]

            pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=pool_size, strides=strides)

            # conv_pool1 = tf.layers.conv3d(inputs=pool1, filters=filter, kernel_size=[3,3], padding='valid',
            #                          activation=activation,
            #                          name=conv_pool_name+self.log_ext, dilation_rate=dilation_rate)
            return pool1,cropped

    # ========================
    def dense_block(self, input,
                    feature_size,
                    is_training_bn,
                    padding='same',
                    activation=None,
                    name='dense_sub_block',
                    flag=0,
                    concat_flag=0,
                    bn_flag=False,
                    dilation_rate=1):

        if bn_flag==False:
            with tf.name_scope(name):
                db_conv1 = tf.layers.conv3d(input,
                                            filters=feature_size[0] * 4,
                                            kernel_size=self.kernel_size1,
                                            padding=padding,
                                            activation=activation)


                db_conv2 = tf.layers.conv3d(db_conv1,
                                            filters=feature_size[1],
                                            kernel_size=self.kernel_size2,
                                            padding=padding,
                                            activation=activation)
        else:
            with tf.name_scope(name):
                db_conv1 = tf.layers.conv3d(input,
                                            filters=feature_size[0] * 4,
                                            kernel_size=self.kernel_size1,
                                            padding=padding,
                                            activation=None)
                bn1 = tf.layers.batch_normalization(db_conv1, training=is_training_bn,renorm=False)
                bn1 = tf.nn.leaky_relu(bn1)

                db_conv2 = tf.layers.conv3d(bn1,
                                            filters=feature_size[1],
                                            kernel_size=self.kernel_size2,
                                            padding=padding,
                                            activation=None,
                                            dilation_rate=1)
                bn2 = tf.layers.batch_normalization(db_conv2, training=is_training_bn,renorm=False)
                bn2 = tf.nn.leaky_relu(bn2)
                db_conv2=bn2

        db_concat = tf.concat([input, db_conv2], 4)
        return db_concat



    # ========================
    def dense_loop(self, loop, input, crop_size,
                   db_size,is_training_bn,
                   padding='same',
                   activation=None,
                   name='dense_block',
                   flag=0,
                   concat_flag=0,
                   feature_size=[],bn_flag=False,dilation_rate=1):
        with tf.name_scope(name):
            output = input
            for i in range(loop):
                output = self.dense_block(output,
                                          feature_size=feature_size,
                                          padding=padding,
                                          activation=activation,
                                          name='dense_sub_block' +self.log_ext+ str(i),
                                          flag=flag,
                                          concat_flag=concat_flag,
                                          is_training_bn=is_training_bn,
                                          bn_flag=bn_flag,
                                          dilation_rate=dilation_rate)

        cropped = output[:,
                  tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                  tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2),
                  tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                  tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2),
                  tf.to_int32(db_size / 2) - tf.to_int32(crop_size / 2) - 1:
                  tf.to_int32(db_size / 2) + tf.to_int32(crop_size / 2), :]
        # #
        # cropped = output[:,
        #           np.int32(db_size / 2) - np.int32(crop_size / 2) - 1:
        #           np.int32(db_size / 2) + np.int32(crop_size / 2),
        #           np.int32(db_size / 2) - np.int32(crop_size / 2) - 1:
        #           np.int32(db_size / 2) + np.int32(crop_size / 2),
        #           np.int32(db_size / 2) - np.int32(crop_size / 2) - 1:
        #           np.int32(db_size / 2) + np.int32(crop_size / 2), :]
        return output, cropped


    # ========================

    def dens_net(self, image, is_training, dropout_rate1,dropout_rate2, dim,is_training_bn):
        # x = 527
        # l2 = x / 2
        # l3 = l2 / 2
        # l4 = l3 / 2
        #
        # l4_1 = l4 - 2
        #
        # dl4 = int(l4_1) * 2 + 1
        # dl4_1 = dl4 - 2
        # dl3 = dl4_1 * 2 + 1
        # dl3_1 = dl3 - 2
        #
        # dl2 = dl3_1 * 2 + 1
        # dl2_1 = dl2 - 2

        # dim2 = 73
        # db_size1 = np.int32(dim2)
        # db_size2 = np.int32(db_size1 / 2)
        # db_size3 = np.int32(db_size2 / 2)
        # crop_size1 = np.int32(((db_size3 - 2) * 2 + 1.0))
        # crop_size2 = np.int32((crop_size1 - 2) * 2 + 1)
        # db_size0 = 0
        # crop_size0 = 0

        db_size1 = tf.to_int32(dim)
        db_size2 = tf.to_int32(db_size1 / 2)
        db_size3 = tf.to_int32(db_size2 / 2)
        crop_size1 = tf.add(tf.multiply(db_size3-2, 2), 1)
        crop_size2 = tf.add(tf.multiply(tf.add(crop_size1, -2), 2), 1)
        db_size0 = tf.to_int32(0)
        crop_size0 = tf.to_int32(0)

        activation=tf.nn.leaky_relu

        self.seed+=1
        with tf.Session() as s:
            rnd = s.run(tf.random_uniform([1], 0, 5, dtype=tf.int32,seed=self.seed))#, seed=int(time.time())))


        noisy_img = tf.cond(is_training,
                            lambda: image + tf.round(tf.random_normal(tf.shape(image), mean=0,
                                                                      stddev=rnd,
                                                                      seed=self.seed+2,#int(time.time()),
                                                                      dtype=tf.float32))
                            , lambda: image)


        conv0 = tf.layers.conv3d(inputs=noisy_img, filters=16, kernel_size=3,
                                 padding='same',
                                 activation=None,
                                 name='conv_deconv_0' + self.log_ext,
                                 dilation_rate=(1, 1, 1))
        bn1 = tf.layers.batch_normalization(conv0, training=is_training_bn,renorm=False)
        bn1 = tf.nn.leaky_relu(bn1)
        bn1 = tf.concat([noisy_img, bn1], 4)


        conv1 = tf.layers.conv3d(inputs=bn1, filters=16, kernel_size=3,
                                 padding='same',
                                 activation=None,
                                 name='conv_deconv_00' + self.log_ext,
                                 dilation_rate=(1, 1, 1))
        bn2 = tf.layers.batch_normalization(conv1, training=is_training_bn,renorm=False)
        bn2 = tf.nn.leaky_relu(bn2)

        bn2 = tf.concat([noisy_img, bn2], 4)

        conv2 = tf.layers.conv3d(inputs=bn2, filters=16, kernel_size=3,
                                 padding='same',
                                 activation=None,
                                 name='conv_deconv_000' + self.log_ext,
                                 dilation_rate=(1, 1, 1))
        bn3 = tf.layers.batch_normalization(conv2, training=is_training_bn, renorm=False)
        bn3 = tf.nn.leaky_relu(bn3)

        bn3= tf.concat([noisy_img, bn3], 4)


        [dense_out1, conc1] = self.dense_loop(loop=self.config[0],
                                              input=bn3,
                                              crop_size=crop_size2,
                                              db_size=db_size1,
                                              padding='same',
                                              activation=activation,
                                              name='dense_block_1'+self.log_ext,
                                              concat_flag=1,
                                              feature_size=[8,8],
                                              is_training_bn=is_training_bn,
                                              bn_flag=True,
                                              dilation_rate=1)



        [pool1, conc1] = self.transition_layer(dense_out1, 'transition_1',
                                      conv_name='conv1'+self.log_ext,
                                      conv_pool_name='conv_pool_name1'+self.log_ext,
                                      db_size=db_size1, crop_size=crop_size2,
                                      kernel_size=[1, 1, 1], padding='same',
                                      activation=activation,
                                      dilation_rate=(1, 1, 1),
                                      pool_size=[2, 2, 2],
                                      strides=(2,2,2),
                                      is_training_bn=is_training_bn,
                                      bn_flag=True)


        # ========================
        [dense_out2, conc2] = self.dense_loop(loop=self.config[1],
                                              input=pool1,
                                              crop_size=crop_size1,
                                              db_size=db_size2,
                                              padding='same',
                                              activation=activation,
                                              name='dense_block_2'+self.log_ext,
                                              feature_size=[8,8],
                                              is_training_bn=is_training_bn,
                                              bn_flag=True,
                                              dilation_rate=1)


        [pool2,conc2] = self.transition_layer(dense_out2, 'transition_2',
                                      conv_name='conv2'+self.log_ext,
                                      conv_pool_name='conv_pool_name2'+self.log_ext,
                                      db_size=db_size2, crop_size=crop_size1,
                                      kernel_size=[1, 1, 1],
                                      padding='same',
                                      activation=activation,
                                      dilation_rate=(1, 1, 1),
                                      pool_size=[2, 2, 2],
                                      strides=(2,2,2),
                                      is_training_bn=is_training_bn,
                                      bn_flag=True)
        # ========================
        [dense_out3, conc3] = self.dense_loop(loop=self.config[2],
                                              input=pool2,
                                              crop_size=crop_size0,
                                              db_size=db_size0,
                                              padding='same',
                                              activation=activation,
                                              name='dense_block_3'+self.log_ext,
                                              feature_size=[8,8]
                                              , is_training_bn=is_training_bn,
                                              bn_flag=True,
                                              dilation_rate=1)



        conv1 = tf.layers.conv3d(inputs=dense_out3,
                                 filters=int(dense_out3.shape[4].value),
                                 kernel_size=[3, 3, 3],
                                 padding='valid',
                                 activation=None,
                                 name='conv_deconv_1'+self.log_ext,
                                 dilation_rate=(1, 1, 1))

        bn2 = tf.layers.batch_normalization(conv1, training=is_training_bn,renorm=False)
        bn2 = tf.nn.leaky_relu(bn2)



        # ========================
        deconv1 = tf.layers.conv3d_transpose(bn2,
                                             filters=int(conv1.shape[4].value/2),
                                             kernel_size=3,
                                             strides=(2, 2, 2),
                                             padding='valid',
                                             use_bias=False)
        conc11=tf.concat([conc2, deconv1], 4)


        [dense_out5, conctmp] = self.dense_loop(loop=self.config[3],
                                                input=conc11,
                                                crop_size=crop_size0,
                                                db_size=db_size0,
                                                padding='same',
                                                activation=activation,
                                                name='dense_block_5'+self.log_ext,
                                                feature_size=[8,8]
                                                , is_training_bn=is_training_bn,
                                              bn_flag=True,
                                                dilation_rate=1)


        conv2 = tf.layers.conv3d(inputs=dense_out5,
                                 filters=int(dense_out5.shape[4].value/2),
                                 kernel_size=[3, 3, 3],
                                 padding='valid',
                                 activation=None,
                                 name='conv_deconv_2'+self.log_ext,
                                 dilation_rate=(1, 1, 1))
        bn3 = tf.layers.batch_normalization(conv2, training=is_training_bn,renorm=False)
        bn3 = tf.nn.leaky_relu(bn3)
        # bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training_bn,renorm=False)
        # bn_conv2 = tf.nn.leaky_relu(bn_conv2)
        # =========================================================
        deconv2 = tf.layers.conv3d_transpose(bn3, filters=int(conv2.shape[4].value/2), kernel_size=[3, 3, 3], strides=(2, 2, 2),
                                             padding='valid', use_bias=False)
        conc22 = tf.concat([conc1, deconv2], 4)
        [dense_out6, conctmp] = self.dense_loop(loop=self.config[4],
                                                input=conc22,
                                                crop_size=crop_size0,
                                                db_size=db_size0,
                                                padding='same',
                                                activation=activation,
                                                name='dense_block_6'+self.log_ext,
                                                feature_size=[8,8],
                                                is_training_bn=is_training_bn,
                                              bn_flag=True,
                                                dilation_rate=1)

        conv3 = tf.layers.conv3d(inputs=dense_out6,
                                 filters=int(dense_out6.shape[4].value / 2),
                                 kernel_size=[3, 3, 3],
                                 padding='valid',
                                 activation=None,
                                 name='conv_deconv_tmp' + self.log_ext,
                                 dilation_rate=(1, 1, 1))
        bn4 = tf.layers.batch_normalization(conv3, training=is_training_bn,renorm=False)
        bn4 = tf.nn.leaky_relu(bn4)
        # bn_conv3=tf.layers.batch_normalization(conv3,training=is_training_bn,renorm=False)
        # bn_conv3=tf.nn.leaky_relu(bn_conv3)

        # =========================================================


        # classification layer:
        with tf.name_scope('classification_layer'):
            # post_rm = tf.reduce_mean(pool3, [1, 2], name='global_avg_pool'+self.log_ext, keep_dims=True)
            # fc1 = tf.layers.conv3d(conv3, filters=32, kernel_size=[3,3,3], padding='same', strides=(1, 1, 1),
            #                      activation=activation, dilation_rate=(1, 1,1),name='fc1'+self.log_ext)

            # dropout1 = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=is_training,name='droup_out1'+self.log_ext)

            # fc2 = tf.layers.conv3d(dropout1, filters=128, kernel_size=[3, 3, 3], padding='same', strides=(1, 1, 1),
            #                        activation=tf.nn.leaky_relu, dilation_rate=(1, 1, 1), name='fc2' + self.log_ext)
            #
            # dropout2 = tf.layers.dropout(inputs=fc2, rate=dropout_rate, training=is_training,
            #                             name='droup_out2' + self.log_ext)

            y = tf.layers.conv3d(bn4, filters=self.class_no, kernel_size=1, padding='same', strides=(1, 1, 1),
                                 activation=None, dilation_rate=(1, 1,1), name='fc3'+self.log_ext)

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))




        '''h_fc1 = tf.contrib.layers.fully_connected(post_rm,20,activation_fn=tf.nn.leaky_relu)

        #Fully connected layer #2
        #h_fc2 = tf.layers.conv3d(h_fc1,512,[3,3],padding="valid", strides=(1,1),activation=tf.nn.leaky_relu, dilation_rate=(1, 1))
        h_fc2 = tf.contrib.layers.fully_connected(h_fc1,8,activation_fn=tf.nn.leaky_relu)

        #Fully connected layer #3
        #y =  tf.layers.conv3d(h_fc2,2,[3,3],padding="valid", strides=(1,1),activation=tf.nn.leaky_relu, dilation_rate=(1, 1))
        y = tf.contrib.layers.fully_connected(h_fc2,class_no,activation_fn=tf.nn.leaky_relu)
'''
        return  y#,dense_out1,dense_out2,dense_out3,dense_out5,dense_out6

