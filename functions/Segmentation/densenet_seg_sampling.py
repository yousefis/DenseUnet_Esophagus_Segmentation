
import time
import shutil
import os
# from functions.densenet_unet import _densenet_unet
from functions.networks.dense_unet2 import _densenet_unet
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import logging
# import wandb
# from functions.read_data import _read_data
from functions.data_reader.read_data3 import _read_data
from functions.threads.read_thread_smart_sampling import read_thread
from functions.image_classes.image_class_smart_sampling import image_class
from functions.fill_thread import fill_thread
import functions.settings as settings
from functions.loss_func import _loss_func
import psutil
# calculate the dice coefficient
from shutil import copyfile
from functions.threads.patch_extractor_smart_sampling import _patch_extractor_thread
from functions.optimizer.RectifiedAdam import RAdamOptimizer as RAdam
import random
# --------------------------------------------------------------------------------------------------------
class dense_seg:
    def __init__(self,data,densnet_unet_config ,compression_coefficient ,growth_rate ,
                 sample_no,validation_samples,no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag,img_name,label_name,torso_tag,log_tag,
                 tumor_percent,other_percent,Logs,fold,server_path):
        settings.init()
        random.seed(90)

        # ==================================
        # densenet_unet parameters:
        self.densnet_unet_config = densnet_unet_config
        self.compression_coefficient = compression_coefficient
        self.growth_rate = growth_rate
        # ==================================
        self.train_tag=train_tag
        self.validation_tag=validation_tag
        self.test_tag=test_tag
        self.img_name=img_name
        self.label_name=label_name
        self.torso_tag=torso_tag
        self.data=data
        # ==================================
        self.learning_decay = .95
        self.learning_rate = 1E-5
        self.beta_rate = 0.05

        self.img_padded_size = 519
        self.seg_size = 505

        self.GTV_patchs_size =49# 61#33 #            61
        self.patch_window = 63##87#79#57#47          73
        self.sample_no = sample_no
        self.batch_no = 7
        self.batch_no_validation = 30
        self.validation_samples = validation_samples
        self.display_step = 100
        self.display_validation_step = 1
        self.total_epochs = 10
        self.loss_instance=_loss_func()
        self.server_path= server_path

        self.dropout_keep = .5
        self.no_sample_per_each_itr=no_sample_per_each_itr
        # self.test_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/test/'


        self.log_ext = ''.join(map(str, self.densnet_unet_config))+'_'+str(self.compression_coefficient)+'_'+str(self.growth_rate)+log_tag
        self.LOGDIR = server_path+Logs + self.log_ext + '/'
        self.train_acc_file = server_path+'/Accuracy/' + self.log_ext + '.txt'
        self.train_acc_file = server_path+'/Accuracy/' + self.log_ext + '.txt'
        self.validation_acc_file = server_path+'/Accuracy/' + self.log_ext + '.txt'
        self.chckpnt_dir = server_path+Logs + self.log_ext + '/densenet_unet_checkpoints/'
        self.out_path = server_path+'/Outputs/' + self.log_ext + '/'
        self.x_hist=0

        self.tumor_percent = tumor_percent
        self.other_percent = other_percent
        self.fold=fold

    def copytree(self,src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
    def save_file(self,file_name,txt):
        with open(file_name, 'a') as file:
            file.write(txt)
    def run_net(self):

        # res= tf.test.is_gpu_available(
        #     cuda_only=False, min_cuda_compute_capability=None
        # )
        # if res==False:
        #     print(111111111111111111111111111111111111)
        #     return


        '''read 2d images from the data:'''
        two_dim=True

        _rd = _read_data(data=self.data,train_tag=self.train_tag, validation_tag=self.validation_tag, test_tag=self.test_tag,
                         img_name=self.img_name, label_name=self.label_name,torso_tag=self.torso_tag)
        # _rd = _read_data(train_tag='prostate_train/',validation_tag='prostate_validation/',test_tag='prostate_test/',
        #          img_name='.mha',label_name='Bladder.mha',torso_tag=self.torso_tag)
        # _rd = _read_data(train_tag='train/', validation_tag='validation/', test_tag='test/',
        #                  img_name='CT.mha', label_name='GTV_CT.mha',torso_tag=self.torso_tag)

        flag=False
        self.alpha_coeff=1

        # wandb.init()

        '''read path of the images for train, test, and validation'''

        train_CTs, train_GTVs, train_Torso, train_penalize, \
        validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
        test_CTs, test_GTVs, test_Torso, test_penalize=_rd.read_data_path(fold=self.fold)
        self.img_width = 500
        self.img_height = 500
        # ======================================
        bunch_of_images_no=24
        sample_no=500
        _image_class_vl = image_class(validation_CTs, validation_GTVs, validation_Torso,validation_penalize
                                      , bunch_of_images_no=bunch_of_images_no,  is_training=0,
                                      patch_window=self.patch_window)
        _patch_extractor_thread_vl = _patch_extractor_thread(_image_class=_image_class_vl,
                                                             sample_no=sample_no, patch_window=self.patch_window,
                                                             GTV_patchs_size=self.GTV_patchs_size,
                                                             tumor_percent=self.tumor_percent,
                                                             other_percent=self.other_percent,
                                                             img_no=bunch_of_images_no,
                                                             mutex=settings.mutex,is_training=0,vl_sample_no=self.validation_samples
                                                             )
        _fill_thread_vl = fill_thread(validation_CTs,
                                      validation_GTVs,
                                      validation_Torso,
                                      validation_penalize,
                                      _image_class_vl,
                                      sample_no=sample_no,
                                      total_sample_no=self.validation_samples,
                                      patch_window=self.patch_window,
                                      GTV_patchs_size=self.GTV_patchs_size,
                                      img_width=self.img_width, img_height=self.img_height,
                                      mutex=settings.mutex,
                                      tumor_percent=self.tumor_percent,
                                      other_percent=self.other_percent,
                                      is_training=0,
                                      patch_extractor=_patch_extractor_thread_vl,
                                      fold=self.fold)


        _fill_thread_vl.start()
        _patch_extractor_thread_vl.start()
        _read_thread_vl = read_thread(_fill_thread_vl, mutex=settings.mutex,
                                      validation_sample_no=self.validation_samples, is_training=0)
        _read_thread_vl.start()
        # ======================================
        bunch_of_images_no = 18
        sample_no=140
        _image_class = image_class(train_CTs, train_GTVs, train_Torso,train_penalize
                                   , bunch_of_images_no=bunch_of_images_no,is_training=1,patch_window=self.patch_window
                                   )
        patch_extractor_thread = _patch_extractor_thread(_image_class=_image_class,
                                                         sample_no=sample_no, patch_window=self.patch_window,
                                                         GTV_patchs_size=self.GTV_patchs_size,
                                                         tumor_percent=self.tumor_percent,
                                                         other_percent=self.other_percent,
                                                         img_no=bunch_of_images_no,
                                                         mutex=settings.mutex,is_training=1)
        _fill_thread = fill_thread(train_CTs, train_GTVs, train_Torso,train_penalize,
                                   _image_class,
                                   sample_no=sample_no,total_sample_no=self.sample_no,
                                   patch_window=self.patch_window,
                                   GTV_patchs_size=self.GTV_patchs_size,
                                   img_width=self.img_width,
                                   img_height=self.img_height,mutex=settings.mutex,
                                   tumor_percent=self.tumor_percent,
                                   other_percent=self.other_percent,is_training=1,
                                   patch_extractor=patch_extractor_thread,
                                   fold=self.fold)

        _fill_thread.start()
        patch_extractor_thread.start()

        _read_thread = read_thread(_fill_thread,mutex=settings.mutex,is_training=1)
        _read_thread.start()
        # ======================================
        # pre_bn=tf.placeholder(tf.float32,shape=[None,None,None,None,None])
        # image=tf.placeholder(tf.float32,shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window,1])
        # label=tf.placeholder(tf.float32,shape=[self.batch_no_validation,self.GTV_patchs_size,self.GTV_patchs_size,self.GTV_patchs_size,2])
        # loss_coef=tf.placeholder(tf.float32,shape=[self.batch_no_validation,1,1,1])

        image = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label = tf.placeholder(tf.float32, shape=[None, None, None, None, 2])
        penalize = tf.placeholder(tf.float32, shape=[None, None, None, None,1])
        # loss_coef = tf.placeholder(tf.float32, shape=[None, 2]) # shape: batchno * 2 values for each class
        alpha = tf.placeholder(tf.float32, name='alpha') # background coeff
        beta = tf.placeholder(tf.float32, name='beta') # tumor coeff



        ave_vali_acc=tf.placeholder(tf.float32)
        ave_loss_vali=tf.placeholder(tf.float32)
        ave_dsc_vali=tf.placeholder(tf.float32)

        dropout=tf.placeholder(tf.float32,name='dropout')
        is_training = tf.placeholder(tf.bool, name='is_training')
        is_training_bn = tf.placeholder(tf.bool, name='is_training_bn')
        dense_net_dim = tf.placeholder(tf.int32, name='dense_net_dim')

        # _u_net=_unet()
        # _u_net.unet(image)
        _dn = _densenet_unet(self.densnet_unet_config,self.compression_coefficient,self.growth_rate) #create object
        y=_dn.dens_net(image=image,is_training=is_training,dropout_rate1=0,dropout_rate2=0,dim=dense_net_dim,is_training_bn=is_training_bn)
        # y = _dn.vgg(image)

        y_dirX = ((y[:, int(self.GTV_patchs_size / 2), :, :, 0, np.newaxis]))
        label_dirX = (label[:, int(self.GTV_patchs_size / 2), :, :, 0, np.newaxis])
        penalize_dirX =   (penalize[:,16,:,:,0,np.newaxis])
        image_dirX = ((image[:, int(self.patch_window / 2), :, :, 0, np.newaxis]))
        # x_Fixed = label[0,np.newaxis,:,:,0,np.newaxis]#tf.expand_dims(tf.expand_dims(y[0,10, :, :, 1], 0), -1)
        # x_Deformed = tf.expand_dims(tf.expand_dims(y[0,10, :, :, 1], 0), -1)


        show_img=tf.nn.softmax(y)[:, int(self.GTV_patchs_size / 2) , :, :, 0, np.newaxis]
        tf.summary.image('outprunut',show_img  , 3)
        tf.summary.image('output without softmax',y_dirX ,3)
        tf.summary.image('groundtruth', label_dirX,3)
        tf.summary.image('penalize', penalize_dirX,3)
        tf.summary.image('image',image_dirX ,3)

        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # with tf.Session() as sess:
        devices = sess.list_devices()
        print(devices)
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')

        log_extttt=''#self.log_ext.split('_')[0]+'01'
        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' + log_extttt,graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' + log_extttt, graph=sess.graph)

        # y=_dn.vgg(image)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        saver=tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        loadModel = 0
        # if loadModel==0:

        # self.copytree('',self.LOGDIR +'')
        try:
            self.copytree('./functions/', self.LOGDIR + '/functions/')
            self.copytree('./runers/', self.LOGDIR + '/runers/')
            copyfile('./run_allnet_miccai.py', self.LOGDIR + 'run_allnet_miccai.py')
            copyfile('./run_allnets.py', self.LOGDIR + 'run_allnets.py')
            copyfile('./run_attention_job.py', self.LOGDIR + 'run_attention_job.py')
            copyfile('./run_attentionnet.py', self.LOGDIR + 'run_attentionnet.py')
            copyfile('./run_job.py', self.LOGDIR + 'run_job.py')
            copyfile('./run_net_sampling.py', self.LOGDIR + 'run_net_sampling.py')
            copyfile('./setup.py', self.LOGDIR + 'setup.py')
            copyfile('./submit_sampling.py', self.LOGDIR + 'submit_sampling.py')
            copyfile('./submit_test_job.py', self.LOGDIR + 'submit_test_job.py')
            copyfile('./test_all_densenet_unet.py', self.LOGDIR + 'test_all_densenet_unet.py')
            copyfile('./test_deneunet_miccai.py', self.LOGDIR + 'test_deneunet_miccai.py')
            copyfile('./test_densenet_unet.py', self.LOGDIR + 'test_densenet_unet.py')
            copyfile('./test_job.sh', self.LOGDIR + 'test_job.sh')
            copyfile('./test_network_newdata.py', self.LOGDIR + 'test_network_newdata.py')
        except:
            print('File exists?')


        '''AdamOptimizer:'''
        with tf.name_scope('cost'):
            [ penalized_loss,
             soft_dice_coef,logt,lbl]=self.loss_instance.dice_plus_distance_penalize(logits=y, labels=label,penalize=penalize)
            cost = tf.reduce_mean((1.0 - soft_dice_coef[1]), name="cost")



        tf.summary.scalar("cost", cost)
        # tf.summary.scalar("cost_before", cost_before)
        # tf.summary.scalar("recall_precision", recall_precision)
        f1_measure = self.loss_instance.f1_measure(logits=y, labels=label)
        tf.summary.scalar("dice_bakground", f1_measure[0])
        tf.summary.scalar("dice_tumor", f1_measure[1])

        pwc = self.loss_instance.PWC(y, label)
        tf.summary.scalar("pwc_bakground", pwc[0])
        tf.summary.scalar("pwc_tumor", pwc[1])

        recall = self.loss_instance.Recall(y, label)
        tf.summary.scalar("recall_bakground", recall[0])
        tf.summary.scalar("recall_tumor", recall[1])

        precision = self.loss_instance.Precision(y, label)
        tf.summary.scalar("precision_bakground", precision[0])
        tf.summary.scalar("precision_tumor", precision[1])

        fpr = self.loss_instance.FPR(y, label)
        tf.summary.scalar("FPR_bakground", fpr[0])
        tf.summary.scalar("FPR_tumor", fpr[1])

        fnr = self.loss_instance.FNR(y, label)
        tf.summary.scalar("FNR_bakground", fnr[0])
        tf.summary.scalar("FNR_tumor", fnr[1])

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer_tmp = tf.train.AdamOptimizer(self.learning_rate)
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # optimizer = RAdam(learning_rate=self.learning_rate)
            # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_tmp)
            optimizer = optimizer_tmp.minimize(cost)

        with tf.name_scope('validation'):
            average_validation_accuracy=ave_vali_acc
            average_validation_loss=ave_loss_vali
            average_dsc_loss=ave_dsc_vali
        tf.summary.scalar("average_validation_accuracy",average_validation_accuracy)
        tf.summary.scalar("average_validation_loss",average_validation_loss)
        tf.summary.scalar("average_dsc_loss",average_dsc_loss)

        with tf.name_scope('accuracy'):
            # accuracy=dsc_fn(y, label,1E-4)
            accuracy=self.loss_instance.accuracy_fn(y, label)
            # f1_score=self.loss_instance.f1_measure(y, label)
            # accuracy=tf.reduce_mean(f1_score)
        tf.summary.scalar("accuracy", accuracy)
        # tf.summary.scalar("f1_score1",f1_score[0])
        # tf.summary.scalar("f1_score2",f1_score[1])

        # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train_writer.add_graph(sess.graph)
        logging.debug('total number of variables %s' % (
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        summ=tf.summary.merge_all()

        point = 0
        itr1 = 0
        if loadModel:
            chckpnt_dir='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/dilated-mid/22322_1_4-without-firstlayers-01/densenet_unet_checkpoints/'
            ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
            # with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            point=np.int16(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            itr1=point


        # patch_radius = 49
        '''loop for epochs'''

        for epoch in range(self.total_epochs):
            while self.no_sample_per_each_itr*int(point/self.no_sample_per_each_itr)<self.sample_no:
                print("epoch #: %d" %(epoch))
                startTime = time.time()

                step = 0

                self.beta_coeff=1+1 * np.exp(-point/2000)
                # =============validation================
                if itr1 % self.display_validation_step ==0:
                    '''Validation: '''
                    loss_validation = 0
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation=0

                    while (validation_step * self.batch_no_validation <settings.validation_totalimg_patch):
                        [validation_CT_image, validation_GTV_image,validation_Penalize_patch] = _image_class_vl.return_patches_validation(validation_step * self.batch_no_validation, (validation_step + 1) *self.batch_no_validation)
                        if (len(validation_CT_image)<self.batch_no_validation) | (len(validation_GTV_image)<self.batch_no_validation) | (len(validation_Penalize_patch)<self.batch_no_validation) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            continue

                        validation_CT_image_patchs = validation_CT_image
                        validation_GTV_label = validation_GTV_image
                        tic=time.time()
                        [acc_vali, loss_vali,dsc_vali] = sess.run([accuracy, cost,f1_measure],
                                                         feed_dict={image: validation_CT_image_patchs,
                                                                    label: validation_GTV_label,
                                                                    penalize: validation_Penalize_patch,
                                                                    dropout: 1,
                                                                    is_training: False,
                                                                    ave_vali_acc: -1,
                                                                    ave_loss_vali: -1,
                                                                    ave_dsc_vali:-1,
                                                                    dense_net_dim: self.patch_window,
                                                                    is_training_bn:False,
                                                                    alpha:1,
                                                                    beta:1})


                        elapsed=time.time()-tic

                        acc_validation += acc_vali
                        loss_validation += loss_vali
                        dsc_validation+=dsc_vali[1]
                        validation_step += 1
                        if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                            print('nan problem')
                        process = psutil.Process(os.getpid())
                        print(
                            '%d - > %d: elapsed_time:%d acc_validation: %f, loss_validation: %f, memory_percent: %4s' % (
                                validation_step,validation_step * self.batch_no_validation
                                , elapsed, acc_vali, loss_vali, str(process.memory_percent()),
                            ))

                    settings.queue_isready_vl = False
                    acc_validation = acc_validation / (validation_step)
                    loss_validation = loss_validation / (validation_step)
                    dsc_validation = dsc_validation / (validation_step)
                    if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                        print('nan problem')
                    _fill_thread_vl.kill_thread()
                    print('******Validation, step: %d , accuracy: %.4f, loss: %f*******' % (
                    itr1, acc_validation, loss_validation))
                    [sum_validation] = sess.run([summ],
                                                feed_dict={image: validation_CT_image_patchs,
                                                           label: validation_GTV_label,
                                                           penalize: validation_Penalize_patch,
                                                           dropout: 1,
                                                           is_training: False,
                                                           ave_vali_acc: acc_validation,
                                                           ave_loss_vali: loss_validation,
                                                           ave_dsc_vali:dsc_validation,
                                                           dense_net_dim: self.patch_window,
                                                           is_training_bn: False,
                                                           alpha: 1,
                                                           beta: 1
                                                           })
                    validation_writer.add_summary(sum_validation, point)
                    print('end of validation---------%d' % (point))

                '''loop for training batches'''
                while(step*self.batch_no<self.no_sample_per_each_itr):

                    [train_CT_image_patchs, train_GTV_label,
                     train_Penalize_patch,loss_coef_weights] = _image_class.return_patches( self.batch_no,point)


                    if (len(train_CT_image_patchs)<self.batch_no)|(len(train_GTV_label)<self.batch_no)\
                            |(len(train_Penalize_patch)<self.batch_no):
                        #|(len(train_Penalize_patch)<self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue


                    tic=time.time()
                    # with tf.Session() as sess:
                    [acc_train1, loss_train1, optimizing,out,dsc_train11] = sess.run([accuracy, cost, optimizer,y,f1_measure],
                                                                     feed_dict={image: train_CT_image_patchs,
                                                                                label: train_GTV_label,
                                                                                penalize: train_Penalize_patch,
                                                                                # loss_coef: loss_coef_weights,
                                                                                dropout: self.dropout_keep,
                                                                                is_training: True,
                                                                                ave_vali_acc: -1,
                                                                                ave_loss_vali: -1,
                                                                                ave_dsc_vali: -1,
                                                                                dense_net_dim: self.patch_window,
                                                                                is_training_bn: True,
                                                                                alpha: self.alpha_coeff,
                                                                                beta: self.beta_coeff
                                                                                })

                    settings.patch_list.append(loss_train1,train_CT_image_patchs, train_GTV_label, train_Penalize_patch)
                    refine_counter=20
                    if point % refine_counter == 0: # every refine_counter%50 iteration keep the worse batches in the list
                        print('<Refine! %s',str(process.memory_percent()))
                        settings.patch_list.refine()
                        print('Refine! %s>',str(process.memory_percent()))

                    mutation_counter =40
                    if point % mutation_counter == 0:
                        print('<intercourse!')
                        settings.patch_list.intercourse()
                        strong_child_CT=[]
                        strong_child_GTV=[]
                        strong_child_Penalize=[]
                        if point:
                            print('replacement!%s ',str(process.memory_percent()))
                            for ch in range(len(settings.patch_list.children)):
                                [loss_samples] = sess.run(
                                    [cost],
                                    feed_dict={image: settings.patch_list.children[ch][1][0],
                                               label: settings.patch_list.children[ch][1][1],
                                               penalize: settings.patch_list.children[ch][1][2],
                                               dropout: self.dropout_keep,
                                               is_training: True,
                                               ave_vali_acc: -1,
                                               ave_loss_vali: -1,
                                               ave_dsc_vali: -1,
                                               dense_net_dim: self.patch_window,
                                               is_training_bn: True,
                                           alpha: self.alpha_coeff,
                                           beta: self.beta_coeff
                                           })
                                if loss_samples>settings.patch_list.worst_patch_list[settings.patch_list.children[ch][0][0]][-1]:
                                    # settings.patch_list.worst_patch_list[ch][0][0] = settings.patch_list.children[ch] #replace parent with child
                                    if len(strong_child_CT)==0:
                                        strong_child_CT=np.squeeze(settings.patch_list.children[ch][1][0])
                                        strong_child_GTV=np.squeeze(settings.patch_list.children[ch][1][1])
                                        strong_child_Penalize=np.squeeze(settings.patch_list.children[ch][1][2])
                                    else:
                                        strong_child_CT=np.vstack((strong_child_CT,np.squeeze(settings.patch_list.children[ch][1][0])))
                                        strong_child_GTV=np.vstack((strong_child_GTV,np.squeeze(settings.patch_list.children[ch][1][1])))
                                        strong_child_Penalize=np.vstack((strong_child_Penalize,np.squeeze(settings.patch_list.children[ch][1][2])))
                                [loss_samples] = sess.run(
                                    [cost],
                                    feed_dict={image: settings.patch_list.children[ch][2][0],
                                               label: settings.patch_list.children[ch][2][1],
                                               penalize: settings.patch_list.children[ch][2][2],
                                               dropout: self.dropout_keep,
                                               is_training: True,
                                               ave_vali_acc: -1,
                                               ave_loss_vali: -1,
                                               ave_dsc_vali: -1,
                                               dense_net_dim: self.patch_window,
                                               is_training_bn: True,
                                               alpha: self.alpha_coeff,
                                               beta: self.beta_coeff
                                               })
                                if loss_samples>settings.patch_list.worst_patch_list[settings.patch_list.children[ch][0][1]][-1]:
                                    # settings.patch_list.worst_patch_list[ch][0][0] = settings.patch_list.children[ch] #replace parent with child
                                    if len(strong_child_CT)==0:
                                        strong_child_CT=np.squeeze(settings.patch_list.children[ch][2][0])
                                        strong_child_GTV=np.squeeze(settings.patch_list.children[ch][2][1])
                                        strong_child_Penalize=np.squeeze(settings.patch_list.children[ch][2][2])
                                    else:
                                        strong_child_CT=np.vstack((strong_child_CT,np.squeeze(settings.patch_list.children[ch][2][0])))
                                        strong_child_GTV=np.vstack((strong_child_GTV,np.squeeze(settings.patch_list.children[ch][2][1])))
                                        strong_child_Penalize=np.vstack((strong_child_Penalize,np.squeeze(settings.patch_list.children[ch][2][2])))
                                        print('strong_child_Penalize len: ' + str(len(strong_child_Penalize)))
                            try:

                                print('<list len: ' + str(len(settings.bunch_CT_patches)))
                                print('train_queue.acquire')
                                settings.train_queue.acquire()
                                settings.bunch_CT_patches = np.vstack((settings.bunch_CT_patches,strong_child_CT))
                                settings.bunch_GTV_patches = np.vstack((settings.bunch_GTV_patches,strong_child_GTV[:,:,:,:,1].astype(np.uint8)))
                                settings.bunch_Penalize_patches = np.vstack((settings.bunch_Penalize_patches,strong_child_Penalize))
                                settings.train_queue.release()
                                print('train_queue.release')
                                print('list len: '+str(len(settings.bunch_CT_patches))+'>')
                            except:
                                o=1
                            strong_child_CT=None
                            strong_child_GTV=None
                            strong_child_Penalize=None
                            settings.patch_list.clear_lists()
                        print('intercourse!> %s',str(process.memory_percent()))
                    elapsed=time.time()-tic



                    dsc_train1=dsc_train11[1]

                    self.x_hist=self.x_hist+1

                    [sum_train] = sess.run([summ],
                                           feed_dict={image: train_CT_image_patchs,
                                                      label: train_GTV_label,
                                                      penalize: train_Penalize_patch,
                                                      # loss_coef: loss_coef_weights,
                                                      dropout: self.dropout_keep, is_training: True,
                                                      ave_vali_acc: acc_train1,
                                                      ave_loss_vali: loss_train1,
                                                      ave_dsc_vali: dsc_train1,
                                                      dense_net_dim: self.patch_window,
                                                      is_training_bn: True,
                                                      alpha: self.alpha_coeff,
                                                      beta: self.beta_coeff
                                                      })
                    train_writer.add_summary(sum_train,point)
                    step = step + 1

                    process = psutil.Process(os.getpid())

                    print(
                        'point: %d, elapsed_time:%d step*self.batch_no:%f , LR: %.15f, acc_train1:%f, loss_train1:%f,memory_percent: %4s' % (
                        int((point)),elapsed,
                        step * self.batch_no, self.learning_rate, acc_train1, loss_train1,
                        str(process.memory_percent())))

                    # print('------------step:%d'%((self.no_sample_per_each_itr/self.batch_no)*itr1+step))
                    point=int((point))#(self.no_sample_per_each_itr/self.batch_no)*itr1+step

                    if point%100==0:
                        '''saveing model inter epoch'''
                        chckpnt_path = os.path.join(self.chckpnt_dir,
                                                    ('densenet_unet_inter_epoch%d_point%d.ckpt' % (epoch, point)))
                        saver.save(sess, chckpnt_path, global_step=point)
                        # settings.patch_list.refine()


                    itr1 = itr1 + 1
                    point=point+1

            endTime = time.time()

            '''saveing model after each epoch'''
            chckpnt_path = os.path.join(self.chckpnt_dir, 'densenet_unet.ckpt')
            # with tf.Session() as sess:
            saver.save(sess, chckpnt_path, global_step=epoch)


            print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime - startTime))
