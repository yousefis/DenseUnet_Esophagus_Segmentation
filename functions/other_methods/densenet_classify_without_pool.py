import os, time
import tensorflow as tf
import SimpleITK as sitk
#import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt
import datetime
import scipy.misc


from dense_without_pool import _dense_net

from read_data_dense_without_pool import _read_data
import shutil
Exp = 'dense_net'

LOGDIR = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/densenet_Log/'
train_acc_file='./accuracy/train_densenet.txt'
validation_acc_file='./accuracy/validation_densenet.txt'
chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/densenet_Log/densenet_checkpoints/'
def save_file(file_name,txt):
    with open(file_name, 'a') as file:
        file.write(txt)
def run_net(is_training):
    '''read 2d images from the data:'''
    two_dim=True
    _rd = _read_data()
    flag=False
    if os.path.exists(LOGDIR):
        shutil.rmtree(LOGDIR)



    '''read path of the images for train, test, and validation'''
    train_CTs, train_GTVs, train_Len, validation_CTs, validation_GTVs, \
    validation_Len, test_CTs, test_GTVs, test_Len=_rd.read_image_path()

    GTV_patchs_size=101
    patch_window=101
    sample_no=100000 #badan avaz kon!!1000
    batch_no = 50
    batch_no_validation = 1000
    validation_samples = 100000 # badan avaz kon!!1000
    display_step = 100
    run_validation_steps = display_step

    learning_rate=1E-4
    learning_decay=.95

    if two_dim:
        image=tf.placeholder(tf.float32,shape=[None,None,None,1])
        label=tf.placeholder(tf.float32,shape=[None,None,None,2])
        # image=tf.placeholder(tf.float32,shape=[None,patch_window,patch_window,1])
        # image=tf.placeholder(tf.float32,shape=[None,512,512,1])
        # label=tf.placeholder(tf.float32,shape=[None,GTV_patchs_size,GTV_patchs_size,2])
        # label=tf.placeholder(tf.float32,shape=[None,512,512,2])
        ave_vali_acc=tf.placeholder(tf.float32)
        ave_loss_vali=tf.placeholder(tf.float32)

    dropout=tf.placeholder(tf.float32,name='dropout')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dense_net_dim = tf.placeholder(tf.int32, name='dense_net_dim')

    # _u_net=_unet()
    # _u_net.unet(image)

    _dn = _dense_net() #create object
    y=_dn.dens_net(image,is_training,dropout,dense_net_dim)


    sess = tf.Session()
    train_writer = tf.summary.FileWriter(LOGDIR + '/train' + Exp,graph=tf.get_default_graph())
    validation_writer = tf.summary.FileWriter(LOGDIR + '/validation' + Exp, graph=sess.graph)

    # y=_dn.vgg(image)



    saver=tf.train.Saver()

    '''AdamOptimizer:'''
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=label), name="cost")
    tf.summary.scalar("cost", cost)

    # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    with tf.name_scope('validation'):
        average_validation_accuracy=ave_vali_acc
        average_validation_loss=ave_loss_vali
    tf.summary.scalar("average_validation_accuracy",average_validation_accuracy)
    tf.summary.scalar("average_validation_loss",average_validation_loss)


    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,3),tf.argmax(label, 3) )
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy",accuracy)

    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)

    summ=tf.summary.merge_all()


    total_epochs=100
    img_width = 512
    img_height = 512
    # patch_radius = 49
    '''loop for epochs'''
    itr1=0
    for epoch in range(total_epochs):
        print('0')
        save_file(train_acc_file, 'epoch: %d\n' % (epoch))
        save_file(validation_acc_file, 'epoch: %d\n' % (epoch))
        print("epoch #: %d" %(epoch))
        startTime = time.time()
        GTV_patchs, CT_image_patchs=_rd.read_data_all_train_batches(train_CTs,train_GTVs,train_Len,sample_no,GTV_patchs_size,patch_window,img_width,img_height,epoch)

        validation_CT_image,validation_GTV_image=_rd.read_all_validation_batches(validation_CTs,validation_GTVs,validation_Len,
                                                                                validation_samples,patch_window,img_width,img_height,epoch,GTV_patchs_size,whole_image=0)
        step=0

        '''loop for training batches'''
        while(step*batch_no<sample_no):
            # print('Training %d samples of %d' %(step*batch_no,sample_no))

            train_CT_image_patchs=CT_image_patchs[step*batch_no:(step+1)*batch_no-1]
            train_GTV_label=GTV_patchs[step*batch_no:(step+1)*batch_no-1]

            [acc_train1,loss_train1,optimizing]=sess.run([accuracy,cost,optimizer],feed_dict={image: train_CT_image_patchs, label: train_GTV_label,dropout:0.5,is_training:True,
                                                       ave_vali_acc:-1,ave_loss_vali:-1,dense_net_dim:patch_window})
            save_file(train_acc_file, '%f, %f\n' % (acc_train1, loss_train1))

            if itr1%display_step==0:
                [sum_train]=sess.run([summ],
                                                    feed_dict={image: train_CT_image_patchs, label: train_GTV_label,dropout:0.5,is_training:True,
                                                               ave_vali_acc:acc_train1,ave_loss_vali:loss_train1,dense_net_dim:patch_window})
                train_writer.add_summary(sum_train,itr1)

            #=============validation================
                '''Validation: '''
                validation_step = 0
                loss_validation = 0
                acc_validation = 0

                while (validation_step * batch_no_validation < validation_samples):
                    validation_CT_image_patchs=validation_CT_image[validation_step*batch_no_validation:(validation_step+1)*batch_no_validation-1]
                    validation_GTV_label=validation_GTV_image[validation_step*batch_no_validation:(validation_step+1)*batch_no_validation-1]
                    [acc_vali, loss_vali] = sess.run([accuracy, cost],
                                                                     feed_dict={image: validation_CT_image_patchs,
                                                                                label: validation_GTV_label, dropout: 1,
                                                                                is_training: False,ave_vali_acc:-1,ave_loss_vali:-1,dense_net_dim:patch_window})

                    acc_validation += acc_vali
                    loss_validation += loss_vali
                    validation_step += 1
                    # print(validation_step * batch_no_validation )
                    save_file(validation_acc_file, '%f, %f\n' % (acc_vali, loss_vali))
                    # print('Validation, step: %d accuracy: %.4f loss: %f' % (
                    # itr1, acc_vali, loss_vali))
                    # end while

                acc_validation = acc_validation / validation_step
                loss_validation = loss_validation / validation_step
                print('******Validation, step: %d accuracy: %.4f loss: %f*******' % (step,acc_validation, loss_validation))
                [sum_validation] = sess.run([summ],
                                        feed_dict={image: validation_CT_image_patchs,
                                                   label: validation_GTV_label, dropout: 1,
                                                   is_training: False,
                                                   ave_vali_acc: acc_validation,
                                                   ave_loss_vali:loss_validation,dense_net_dim:patch_window})
                validation_writer.add_summary(sum_validation, itr1)
                print('end of validation')
            #end if


            step = step + 1
            itr1 = itr1 + 1
        endTime = time.time()



        print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime-startTime))

        if itr1 % 1 == 0:
            if flag == False:
                img, seg = _rd.read_all_validation_batches(validation_CTs, validation_GTVs, validation_Len,
                                                           validation_samples, patch_window, img_width, img_height,
                                                           epoch+1,
                                                           GTV_patchs_size, whole_image=1)
                scipy.misc.imsave('./dense_out/test.png', img[0][:][:].reshape(511, 511))
                scipy.misc.imsave('./dense_out/label.png', (seg[0][:][:])[:, :, 1])
                flag = True

            [acc_img, loss_img, out] = sess.run([accuracy, cost, y],
                                                            feed_dict={image: img,
                                                                       label: seg, dropout: 1,
                                                                       is_training: False, ave_vali_acc: -1,
                                                                       ave_loss_vali: -1,
                                                                       dense_net_dim: 511})
            # imgplot = plt.imshow((out[0][:][:])[:, :, 1], cmap='gray')
            scipy.misc.imsave('./dense_out/result_%d.png' % (epoch), (out[0][:][:])[:, :, 1])

        '''saveing model after each epoch'''
        chckpnt_path = os.path.join(chckpnt_dir, 'densenet.ckpt')
        saver.save(sess, chckpnt_path, global_step=epoch)


        learning_rate=learning_rate*learning_decay



def main():
    is_training = 1
    run_net(is_training)


if __name__ == '__main__':
    main()
    # imgplot = plt.imshow(_image_patchs[1][:][:], cmap='gray')