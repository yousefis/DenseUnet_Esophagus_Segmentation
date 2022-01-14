import os
import time
import shutil
import psutil
import logging
import numpy as np
import tensorflow as tf
import functions.settings as settings
from functions.loss_func import _loss_func
from functions.read_thread_surface import read_thread
from functions.fill_thread_Surface import fill_thread
from functions.image_class_penalize_surface import image_class
from functions.data_reader.read_data3_surface import _read_data
from functions.patch_extractor_thread import _patch_extractor_thread
from functions.networks.dense_unet2_attention_spatial_skip_ch_attention import _densenet_unet
from tensorflow.python.client import device_lib
# --------------------------------------------------------------------------------------------------------
class dense_seg:
    def __init__(self,data,densnet_unet_config ,compression_coefficient ,growth_rate ,
                 sample_no,validation_samples,no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag,torso_tag,log_tag,
                 tumor_percent,Logs,fold,server_path, learning_decay, learning_rate, beta_rate,
                 img_padded_size, seg_size, GTV_patchs_size, patch_window,batch_no,
                 batch_no_validation, display_step, display_validation_step, total_epochs,dropout_keep,
                 img_size):
        settings.init()

        # densenet_unet parameters
        self.densnet_unet_config = densnet_unet_config
        self.compression_coefficient = compression_coefficient
        self.growth_rate = growth_rate

        self.train_tag=train_tag
        self.validation_tag=validation_tag
        self.test_tag=test_tag
        self.torso_tag=torso_tag
        self.data=data
        self.learning_decay = learning_decay
        self.learning_rate = learning_rate
        self.beta_rate = beta_rate

        self.img_padded_size = img_padded_size
        self.seg_size = seg_size

        self.GTV_patchs_size = GTV_patchs_size  #output patch
        self.patch_window = patch_window #input patch
        self.sample_no = sample_no
        self.batch_no = batch_no
        self.batch_no_validation = batch_no_validation
        self.validation_samples = validation_samples
        self.display_step = display_step
        self.display_validation_step = display_validation_step
        self.total_epochs = total_epochs
        self.loss_instance=_loss_func() #an instance from the class of loss functions
        self.server_path= server_path

        self.dropout_keep = dropout_keep
        self.no_sample_per_each_itr=no_sample_per_each_itr

        self.log_ext = ''.join(map(str, self.densnet_unet_config))+'_'+str(self.compression_coefficient)+'_'+str(self.growth_rate)+log_tag
        self.LOGDIR = server_path+Logs + self.log_ext + '/'
        self.train_acc_file = server_path+'/Accuracy/' + self.log_ext + '.txt'
        self.train_acc_file = server_path+'/Accuracy/' + self.log_ext + '.txt'
        self.validation_acc_file = server_path+'/Accuracy/' + self.log_ext + '.txt'
        self.chckpnt_dir = server_path+Logs + self.log_ext + '/densenet_unet_checkpoints/'
        self.out_path = server_path+'/Outputs/' + self.log_ext + '/'
        self.x_hist=0
        self.img_width = img_size
        self.img_height = img_size

        self.tumor_percent = tumor_percent
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
    def run_net(self,
                pre_trained_chckpnt_dir ='' #for resuming training, load the model from this directory
     ):
        """
        read the list of data & start patch reading threads & start training the network
        :return:
        """

        _rd = _read_data(data=self.data)

        self.alpha_coeff=1

        #read path of the images for train, test, and validation
        train_CTs, train_GTVs, train_Torso, train_penalize, train_surface,\
        validation_CTs, validation_GTVs, validation_Torso, validation_penalize, validation_surface,\
        test_CTs, test_GTVs, test_Torso, test_penalize,test_surface=_rd.read_data_path(fold=self.fold)
        self.img_width = self.img_width
        self.img_height = self.img_height
        # ======================================
        #validation instances
        bunch_of_images_no=20
        _image_class_vl = image_class(validation_CTs, validation_GTVs, validation_Torso,validation_penalize,validation_surface
                                      , bunch_of_images_no=bunch_of_images_no,  is_training=0,
                                      patch_window=self.patch_window)
        _patch_extractor_thread_vl = _patch_extractor_thread(_image_class=_image_class_vl,
                                                             sample_no=self.sample_no, patch_window=self.patch_window,
                                                             GTV_patchs_size=self.GTV_patchs_size,
                                                             tumor_percent=self.tumor_percent,
                                                             img_no=bunch_of_images_no,
                                                             mutex=settings.mutex,is_training=0,vl_sample_no=self.validation_samples
                                                             )
        _fill_thread_vl = fill_thread(validation_CTs,
                                      validation_GTVs,
                                      validation_Torso,
                                      validation_penalize,
                                      validation_surface,
                                      _image_class_vl,
                                      sample_no=self.sample_no,
                                      total_sample_no=self.validation_samples,
                                      patch_window=self.patch_window,
                                      GTV_patchs_size=self.GTV_patchs_size,
                                      img_width=self.img_width, img_height=self.img_height,
                                      mutex=settings.mutex,
                                      tumor_percent=self.tumor_percent,
                                      is_training=0,
                                      patch_extractor=_patch_extractor_thread_vl,
                                      fold=self.fold)


        _fill_thread_vl.start()
        _patch_extractor_thread_vl.start()
        _read_thread_vl = read_thread(_fill_thread_vl, mutex=settings.mutex,
                                      validation_sample_no=self.validation_samples, is_training=0)
        _read_thread_vl.start()
        # ======================================
        #training instances
        bunch_of_images_no = 24
        _image_class = image_class(train_CTs, train_GTVs, train_Torso,train_penalize,train_surface
                                   , bunch_of_images_no=bunch_of_images_no,is_training=1,patch_window=self.patch_window
                                   )
        patch_extractor_thread = _patch_extractor_thread(_image_class=_image_class,
                                                         sample_no=240, patch_window=self.patch_window,
                                                         GTV_patchs_size=self.GTV_patchs_size,
                                                         tumor_percent=self.tumor_percent,
                                                         img_no=bunch_of_images_no,
                                                         mutex=settings.mutex,is_training=1)
        _fill_thread = fill_thread(train_CTs, train_GTVs, train_Torso,train_penalize,train_surface,
                                   _image_class,
                                   sample_no=self.sample_no,total_sample_no=self.sample_no,
                                   patch_window=self.patch_window,
                                   GTV_patchs_size=self.GTV_patchs_size,
                                   img_width=self.img_width,
                                   img_height=self.img_height,mutex=settings.mutex,
                                   tumor_percent=self.tumor_percent,
                                   is_training=1,
                                   patch_extractor=patch_extractor_thread,
                                   fold=self.fold)

        _fill_thread.start()
        patch_extractor_thread.start()

        _read_thread = read_thread(_fill_thread,mutex=settings.mutex,is_training=1)
        _read_thread.start()
        # ======================================

        image = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label = tf.placeholder(tf.float32, shape=[None, None, None, None, 2])
        penalize = tf.placeholder(tf.float32, shape=[None, None, None, None,1])
        surf_map = tf.placeholder(tf.float32, shape=[None, None, None, None,1])
        loss_coef = tf.placeholder(tf.float32, shape=[None, 2]) # shape: batchno * 2 values for each class
        alpha = tf.placeholder(tf.float32, name='alpha') # background coeff
        beta = tf.placeholder(tf.float32, name='beta') # tumor coeff

        ave_vali_acc=tf.placeholder(tf.float32)
        ave_loss_vali=tf.placeholder(tf.float32)
        ave_dsc_vali=tf.placeholder(tf.float32)

        dropout=tf.placeholder(tf.float32,name='dropout')
        is_training = tf.placeholder(tf.bool, name='is_training')
        is_training_bn = tf.placeholder(tf.bool, name='is_training_bn')
        dense_net_dim = tf.placeholder(tf.int32, name='dense_net_dim')

        _dn = _densenet_unet(self.densnet_unet_config,self.compression_coefficient,self.growth_rate) #create object
        y=_dn.dens_net(image=image,is_training=is_training,dropout_rate1=0,dropout_rate2=0,dim=dense_net_dim,is_training_bn=is_training_bn)
        # y = _dn.vgg(image)

        y_dirX = ((y[:, int(self.GTV_patchs_size / 2), :, :, 0, np.newaxis]))
        label_dirX = (label[:, int(self.GTV_patchs_size / 2), :, :, 0, np.newaxis])
        penalize_dirX =   (penalize[:,16,:,:,0,np.newaxis])
        surf_map_dirX =   (surf_map[:,16,:,:,0,np.newaxis])
        image_dirX = ((image[:, int(self.patch_window / 2), :, :, 0, np.newaxis]))

        show_img=tf.nn.softmax(y)[:, int(self.GTV_patchs_size / 2) , :, :, 0, np.newaxis]
        tf.summary.image('outprunut',show_img  , 3)
        tf.summary.image('output without softmax',y_dirX ,3)
        tf.summary.image('groundtruth', label_dirX,3)
        tf.summary.image('penalize', penalize_dirX,3)
        tf.summary.image('surf_map', surf_map_dirX,3)
        tf.summary.image('image',image_dirX ,3)

        print('*****************************************')
        print('*****************************************')
        print('*****************************************')
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        devices = sess.list_devices()
        print(devices)

        print(device_lib.list_local_devices())
        print('*****************************************')
        print('*****************************************')
        print('*****************************************')

        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' ,graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' , graph=sess.graph)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        saver=tf.train.Saver(tf.global_variables(), max_to_keep=1000)



        #define the loss function
        with tf.name_scope('cost'):
            penalize_weight=0
            [ penalized_loss,
             soft_dice_coef,logt,lbl]=self.loss_instance.dice_plus_distance_penalize(logits=y, labels=label,penalize=penalize)
            surface_loss= self.loss_instance.surface_loss(logits=y, labels=label, surf_map=surf_map)
            cost = tf.reduce_mean((1.0 - soft_dice_coef[1])+penalize_weight*penalized_loss+surface_loss, name="cost")

        #Setup the Tensorboard plots
        tf.summary.scalar("cost", cost)
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
            optimizer_tmp = tf.train.AdamOptimizer(self.learning_rate,epsilon=0.001)
            optimizer = optimizer_tmp.minimize(cost)

        with tf.name_scope('validation'):
            average_validation_accuracy=ave_vali_acc
            average_validation_loss=ave_loss_vali
            average_dsc_loss=ave_dsc_vali
        tf.summary.scalar("average_validation_accuracy",average_validation_accuracy)
        tf.summary.scalar("average_validation_loss",average_validation_loss)
        tf.summary.scalar("average_dsc_loss",average_dsc_loss)

        with tf.name_scope('accuracy'):
            accuracy=self.loss_instance.accuracy_fn(y, label)

        tf.summary.scalar("accuracy", accuracy)

        sess.run(tf.global_variables_initializer())
        logging.debug('total number of variables %s' % (
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        summ=tf.summary.merge_all()

        point = 0 # starting point, starts from a value > 0 if training is resumed
        itr1 = 0 # number of iterations
        if len(pre_trained_chckpnt_dir):
            ckpt = tf.train.get_checkpoint_state(pre_trained_chckpnt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            point=int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            itr1=point


        # patch_radius = 49
        '''loop for epochs'''

        for epoch in range(self.total_epochs):
            while self.no_sample_per_each_itr*int(point/self.no_sample_per_each_itr)<self.sample_no:
                print('0')
                print("epoch #: %d" %(epoch))
                startTime = time.time()
                step = 0
                self.beta_coeff=1+1 * np.exp(-point/2000)
                # =============start validation================
                if itr1 % self.display_validation_step ==0:
                    '''Validation: '''
                    loss_validation = 0
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation=0
                    while (validation_step * self.batch_no_validation <settings.validation_totalimg_patch):
                        [validation_CT_image, validation_GTV_image,validation_Penalize_patch,validation_Surface_patch] = _image_class_vl.return_patches_validation( validation_step * self.batch_no_validation, (validation_step + 1) *self.batch_no_validation)
                        if (len(validation_CT_image)<self.batch_no_validation) | (len(validation_GTV_image)<self.batch_no_validation) | (len(validation_Penalize_patch)<self.batch_no_validation)  | (len(validation_Surface_patch)<self.batch_no_validation) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            continue

                        validation_CT_image_patchs = validation_CT_image
                        validation_GTV_label = validation_GTV_image
                        tic=time.time()

                        [acc_vali, loss_vali,dsc_vali,surface_loss1] = sess.run([accuracy, cost,f1_measure,surface_loss],
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
                                                                    beta:1,
                                                                    surf_map:validation_Surface_patch,
                                                                    })
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
                                                           beta: 1,
                                                           surf_map: validation_Surface_patch,

                                                           })
                    validation_writer.add_summary(sum_validation, point)
                    print('end of validation---------%d' % (point))

                #loop for training batches
                while(step*self.batch_no<self.no_sample_per_each_itr):
                    [train_CT_image_patchs, train_GTV_label, train_Penalize_patch,loss_coef_weights,train_Surface_patch] = _image_class.return_patches( self.batch_no)

                    if (len(train_CT_image_patchs)<self.batch_no)|(len(train_GTV_label)<self.batch_no)\
                            |(len(train_Penalize_patch)<self.batch_no)|(len(train_Surface_patch)<self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue

                    tic=time.time()
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
                                                                                beta: self.beta_coeff,
                                                                                surf_map: train_Surface_patch,

                                                                                })
                    elapsed=time.time()-tic
                    dsc_train1=dsc_train11[1]

                    self.x_hist=self.x_hist+1
                    # np.hstack((self.x_hist, [np.ceil(

                    [sum_train] = sess.run([summ],
                                           feed_dict={image: train_CT_image_patchs,
                                                      label: train_GTV_label,
                                                      penalize: train_Penalize_patch,
                                                      dropout: self.dropout_keep, is_training: True,
                                                      ave_vali_acc: acc_train1,
                                                      ave_loss_vali: loss_train1,
                                                      ave_dsc_vali: dsc_train1,
                                                      dense_net_dim: self.patch_window,
                                                      is_training_bn: True,
                                                      alpha: self.alpha_coeff,
                                                      beta: self.beta_coeff,
                                                      surf_map: train_Surface_patch,

                                                      })
                    train_writer.add_summary(sum_train,point)
                    step = step + 1

                    process = psutil.Process(os.getpid())

                    print(
                        'point: %d, elapsed_time:%d step*self.batch_no:%f , LR: %.15f, acc_train1:%f, loss_train1:%f,memory_percent: %4s' % (
                        int((point)),elapsed,
                        step * self.batch_no, self.learning_rate, acc_train1, loss_train1,
                        str(process.memory_percent())))


                    point=int((point))
                    if point%100==0:
                        '''saveing model inter epoch'''
                        chckpnt_path = os.path.join(self.chckpnt_dir,
                                                    ('densenet_unet_inter_epoch%d_point%d.ckpt' % (epoch, point)))
                        saver.save(sess, chckpnt_path, global_step=point)
                    itr1 = itr1 + 1
                    point=point+1
            endTime = time.time()

            #==============
            '''saveing model after each epoch'''
            chckpnt_path = os.path.join(self.chckpnt_dir, 'densenet_unet.ckpt')
            saver.save(sess, chckpnt_path, global_step=epoch)
            print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime - startTime))


