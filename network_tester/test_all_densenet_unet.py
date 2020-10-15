#this is ok
import xlwt
from shutil import copyfile
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
from functions.loss_func import _loss_func
#import math as math
import numpy as np

from functions.networks.densenet_unet import _densenet_unet
from functions.read_data import _read_data
from functions.measurements import _measure
from functions.image_class import image_class

eps = 1E-5
plot_tag = 'densenet'
densnet_unet_config = [1, 3, 3, 3, 1]
ct_cube_size = 255
db_size1 = np.int32(ct_cube_size - 2)
db_size2 = np.int32(db_size1 / 2)
db_size3 = np.int32(db_size2 / 2)
crop_size1 = np.int32(((db_size3 - 2) * 2 + 1.0))
crop_size2 = np.int32((crop_size1 - 2) * 2 + 1)

# db_size1 = np.int32(ct_cube_size-2)
# db_size2 = np.int32(db_size1 / 2)
# db_size3 = np.int32(db_size2 / 2)
# crop_size1 = np.int32(((db_size3 - 2) * 2 + 1.0))
# crop_size2 = np.int32((crop_size1 - 2) * 2 + 1)
gtv_cube_size = 241

gap = ct_cube_size - gtv_cube_size



def dice( im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    print('intersect:%d, sum1:%d, sum2:%d'%(intersection.sum(),im1.sum() , im2.sum()))
    d=2. * intersection.sum() / (im1.sum() + im2.sum() + eps)

    return d
def f1_measure(TP,TN,FP,FN):
    precision=Precision(TP,TN,FP,FN)
    recall=Recall(TP,TN,FP,FN)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # f0:background, f1: tumor
    print(f1)
    return f1

def jaccard( im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Jaccard coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / (im1.sum() + im2.sum() - intersection.sum() + eps)

def Sp(TP,TN,FP,FN):

    Sp=TN/(TN+FP+eps)
    return Sp

def tp_tn_fp_fn(logits, labels):
    y_pred = np.asarray(logits).astype(np.bool)
    y_true = np.asarray(labels).astype(np.bool)
    im1 = y_pred.flatten()
    im2 = y_true.flatten()
    TP_t = len(np.where((im1 == True) & (im2 == True))[0])
    TN_t = len(np.where((im1 == False) & (im2 == False))[0])
    FP_t = len(np.where((im1 == True) & (im2 == False))[0])
    FN_t = len(np.where((im1 == False) & (im2 == True))[0])

    TP_b = len(np.where((im1 == False) & (im2 == False))[0])
    TN_b = len(np.where((im1 == True) & (im2 == True))[0])
    FP_b = len(np.where((im1 == False) & (im2 == True))[0])
    FN_b = len(np.where((im1 == True) & (im2 == False))[0])

    TP=np.array((TP_t,TP_b))

    TN = np.array((TN_t,TN_b))

    FP = np.array((FP_t,FP_b))

    FN = np.array((FN_t,FN_b))

    return TP,TN,FP,FN

def FPR(TP,TN,FP,FN):

    fpr=FP/(TN+FP+eps)
    return fpr

def FNR(TP,TN,FP,FN):
    fnr=FN/(TP+FN+eps)
    return fnr

def Precision(TP,TN,FP,FN):
    precision=TP/(TP+FP+ eps)
    return precision

def Recall(TP,TN,FP,FN):
    recall=TP/(TP+FN+ eps)
    return recall

def compute_dice_jaccard( res, gt):
    im1 = np.asarray(res).astype(np.bool)
    im2 = np.asarray(gt).astype(np.bool)
    im1 = im1.flatten()
    im2 = im2.flatten()
    print(im1.shape)
    print(im2.shape)

    d = (dice(im1, im2))
    j = (jaccard(im1, im2))
    return d, j
def output(filename, sheet, list1, list2, x, y, z):
    book = xlwt.Workbook()
    sh = book.add_sheet(sheet)

    variables = [x, y, z]
    x_desc = 'Display'
    y_desc = 'Dominance'
    z_desc = 'Test'
    desc = [x_desc, y_desc, z_desc]

    col1_name = 'Stimulus Time'
    col2_name = 'Reaction Time'
    n=0
    sh.write(n, 0, col1_name)
    sh.write(n, 1, col2_name)
    #You may need to group the variables together
    #for n, (v_desc, v) in enumerate(zip(desc, variables)):
    for n, v_desc, v in enumerate(zip(desc, variables)):
        sh.write(n, 0, v_desc)
        sh.write(n, 1, v)

    n+=1



    for m, e1 in enumerate(list1, n+1):
        sh.write(m, 0, e1)

    for m, e2 in enumerate(list2, n+1):
        sh.write(m, 1, e2)

    book.save(filename)


def test_all_nets(fold,out_dir,Log):
    # densnet_unet_config = [3, 4, 4, 4, 3]
    compression_coefficient =.75
    growth_rate = 4
    # pad_size = 14
    # ext = ''.join(map(str, densnet_unet_config))  # +'_'+str(compression_coefficient)+'_'+str(growth_rate)
    data=2

    # sample_no=2280000
    # validation_samples=5700
    # no_sample_per_each_itr=3420


    # train_tag='train/'
    # validation_tag='validation/'
    # test_tag='test/'
    # img_name='CT_padded.mha'
    # label_name='GTV_CT_padded.mha'
    # torso_tag='CT_padded_Torso.mha'

    train_tag='train/'
    validation_tag='validation/'
    test_tag='Esophagus/'
    # img_name='CTpadded.mha'
    # label_name='GTV_CTpadded.mha'
    # torso_tag='Torsopadded.mha'

    img_name = ''
    label_name = ''
    torso_tag = ''

    _rd = _read_data(data=data,train_tag=train_tag, validation_tag=validation_tag, test_tag=test_tag,
                             img_name=img_name, label_name=label_name)
    test_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/'+test_tag
    chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+log_tag+'/densenet_unet_checkpoints/'


    # test_CTs, test_GTVs ,test_Torsos= _rd.read_imape_path(test_path)

    train_CTs, train_GTVs, train_Torso, train_penalize, \
    validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
    test_CTs, test_GTVs, test_Torso, test_penalize = _rd.read_data_path(fold=fold)

    # test_CTs=train_CTs
    # test_GTVs=train_GTVs
    # test_Torso=train_Torso
    # test_penalize=train_penalize

    # test_CTs=np.sort(test_CTs)
    # test_GTVs=np.sort(test_GTVs)
    # test_Torso=np.sort(test_Torso)
    # test_penalize=np.sort(test_penalize)
    if test_vali == 1:
        test_CTs = np.sort(validation_CTs)
        test_GTVs = np.sort(validation_GTVs)
        test_Torso = np.sort(validation_Torso)
        test_penalize = np.sort(validation_penalize)
    else:
        test_CTs = np.sort(test_CTs)
        test_GTVs = np.sort(test_GTVs)
        test_Torso = np.sort(test_Torso)
        test_penalize = np.sort(test_penalize)


    lf=_loss_func()
    learning_rate = 1E-4
    # image = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    # label = tf.placeholder(tf.float32, shape=[None, None, None, None, 2])

    image = tf.placeholder(tf.float32, shape=[None, ct_cube_size, ct_cube_size, ct_cube_size, 1])
    label = tf.placeholder(tf.float32, shape=[None, gtv_cube_size, gtv_cube_size, gtv_cube_size, 2])

    ave_vali_acc=tf.placeholder(tf.float32)
    ave_loss_vali=tf.placeholder(tf.float32)

    dropout=tf.placeholder(tf.float32,name='dropout')
    # dropout2=tf.placeholder(tf.float32,name='dropout2')
    is_training = tf.placeholder(tf.bool, name='is_training')
    is_training_bn = tf.placeholder(tf.bool, name='is_training_bn')
    dense_net_dim = tf.placeholder(tf.int32, name='dense_net_dim')
    pnalize = tf.placeholder(tf.float32, shape=[None, None, None, None, 2])
    loss_coef = tf.placeholder(tf.float32, shape=[None, 2]) # shape: batchno * 2 values for each class

    _dn = _densenet_unet(densnet_unet_config, compression_coefficient, growth_rate)  # create object
    dn_out = _dn.dens_net(image, is_training,dropout_rate1=0,dropout_rate2=0, dim=ct_cube_size, is_training_bn=is_training_bn)
    y=tf.nn.softmax(dn_out)
    yyy=tf.nn.log_softmax(dn_out)



    # y=_dn.vgg(image)
    loss_instance=_loss_func()

    accuracy = loss_instance.accuracy_fn(y, label)
    [dice, edited_dice] = loss_instance.penalize_dice(logits=y, labels=label, penalize=pnalize)
    # soft_dice_coef=self.loss_instance.soft_dice(logits=y, labels=label)
    cost = tf.reduce_mean(1.0 - dice[1], name="cost")
    # correct_prediction = tf.equal(tf.argmax(y, 4), tf.argmax(label, 4))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # cost = tf.reduce_mean(lf.tversky(logits=y, labels=label, alpha=0.9, beta=0.1), name="cost")

    # restore the model
    sess = tf.Session()
    saver=tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    _meas = _measure()
    out_path = chckpnt_dir+'output/'
    copyfile('./test_densenet_unet.py',
             '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/' + Log + out_dir + 'test_densenet_unet.py')

    jj = []
    dd = []
    dice_boxplot0 = []
    dice_boxplot1= []
    dice_boxplot = []

    jacc_boxplot = []
    jacc_boxplot0 = []
    jacc_boxplot1 = []

    f1_boxplot0=[]
    f1_boxplot1=[]
    f1_boxplot_av=[]

    fpr_av=[]
    fnr_av=[]
    xtickNames = []
    name_list=[]


    fpr0=[]
    fpr1=[]

    fnr0=[]
    fnr1=[]

    sp0=[]
    sp1=[]

    recall0=[]
    recall1=[]
    recall_av=[]
    presicion0=[]
    presicion1=[]
    presicion_av=[]
    img_class=image_class(test_CTs,test_GTVs,test_Torso,test_penalize
                     ,bunch_of_images_no=20,is_training=1,patch_window=ct_cube_size,gtv_patch_window=gtv_cube_size )



    for img_indx in range(1,len(test_CTs)):
        print('img_indx:%s' %(img_indx))
        ss = str(test_CTs[img_indx]).split("/")
        name = ss[8] + '_' + ss[9]
        name = (ss[8] + '_' + ss[9] + '_' + ss[10].split('%')[0]).split('_CT')[0]
        [CT_image, GTV_image, Torso_image,
         volume_depth, voxel_size, origin, direction] = _rd.read_image_seg_volume(test_CTs,
                                                                                  test_GTVs,
                                                                                  test_Torso,
                                                                                  img_indx,
                                                                                  ct_cube_size,
                                                                                  gtv_cube_size)



        _zz_img_gt=[]
        for _z in (range(int(ct_cube_size / 2) +1, CT_image.shape[0] - int(ct_cube_size / 2)+7, int(ct_cube_size )-int(gap)+1)):
            _xx_img_gt=[]
            for _x in (range(int(ct_cube_size / 2) + 1, CT_image.shape[1] - int(ct_cube_size / 2)+7,int(ct_cube_size )-int(gap)+1)):
                _yy_img_gt = []
                for _y in (range(int(ct_cube_size / 2) + 1, CT_image.shape[2] - int(ct_cube_size / 2)+7, int(ct_cube_size )-int(gap)+1)):



                    ct=CT_image[_z-int(ct_cube_size/2)-1:_z+int(ct_cube_size/2),
                                   _x-int(ct_cube_size/2)-1:_x+int(ct_cube_size/2),
                                   _y-int(ct_cube_size/2)-1:_y+int(ct_cube_size/2)]
                    ct = ct[np.newaxis][..., np.newaxis]
                    gtv = GTV_image[_z - int(gtv_cube_size / 2) - 1:_z + int(gtv_cube_size / 2),
                          _x - int(gtv_cube_size / 2) - 1:_x + int(gtv_cube_size / 2),
                          _y - int(gtv_cube_size / 2)-1:_y+int(gtv_cube_size / 2) ]


                    gtv=np.int32(gtv/np.max(GTV_image))

                    gtv = np.eye(2)[gtv]
                    gtv = gtv[np.newaxis]

                    if len(np.where(gtv[0,:,:,:,1]!=0)[0]):
                        print('o')

                    [acc_vali, loss_vali,out,dn_out1] = sess.run([accuracy, cost,y,yyy],
                                                         feed_dict={image: ct,
                                                                    label: gtv,
                                                                    # pnalize: pnlz,
                                                                    # loss_coef:loss_coef_weights,
                                                                    dropout: 1,
                                                                    is_training: False,
                                                                    ave_vali_acc: -1,
                                                                    ave_loss_vali: -1,
                                                                    dense_net_dim: ct_cube_size,
                                                                    is_training_bn: False,

                                                                    })

                    if len(_yy_img_gt)==0:
                        _yy_img_gt = np.int32(gtv[0, :, :, :, 1])
                        _yy_img = np.int32(out[0, :, :, :, 1])

                        _yy_img_ct = CT_image[_z - int(gtv_cube_size / 2) - 1 :_z + int(
                                    gtv_cube_size / 2) ,
                                         _x - int(gtv_cube_size / 2) - 1 :_x + int(
                                             gtv_cube_size / 2) ,
                                         _y - int(gtv_cube_size / 2) - 1 :_y + int(
                                             gtv_cube_size / 2) ]
                    else:
                        _yy_img_gt = np.concatenate((_yy_img_gt, gtv[0, :, :, :, 1]), axis=2)
                        _yy_img = np.concatenate((_yy_img, out[0, :, :, :, 1]), axis=2)

                        _yy_img_ct=np.concatenate((_yy_img_ct, CT_image[_z - int(gtv_cube_size / 2) - 1 :_z + int(
                            gtv_cube_size / 2) ,
                                 _x - int(gtv_cube_size / 2) - 1 :_x + int(
                                     gtv_cube_size / 2) ,
                                 _y - int(gtv_cube_size / 2) - 1 :_y + int(
                                     gtv_cube_size / 2) ]), axis=2)


                if len(_xx_img_gt)==0:
                    _xx_img_gt=_yy_img_gt
                    _xx_img=_yy_img
                    _xx_img_ct=_yy_img_ct
                else:
                    _xx_img_gt = np.concatenate((_xx_img_gt, _yy_img_gt), axis=1)
                    _xx_img = np.concatenate((_xx_img, _yy_img), axis=1)
                    _xx_img_ct = np.concatenate((_xx_img_ct, _yy_img_ct), axis=1)

            if len(_zz_img_gt)==0:
                _zz_img_gt=_xx_img_gt
                _zz_img=_xx_img
                _zz_img_ct=_xx_img_ct
            else:
                _zz_img_gt = np.concatenate((_zz_img_gt, _xx_img_gt), axis=0)
                _zz_img = np.concatenate((_zz_img, _xx_img), axis=0)
                _zz_img_ct = np.concatenate((_zz_img_ct, _xx_img_ct), axis=0)

        name_list.append(name)

        #
        [TP,TN,FP,FN]=tp_tn_fp_fn(np.round(_zz_img), _zz_img_gt)

        f1=f1_measure(TP,TN,FP,FN)
        print('%s: f1:%f,f1:%f'%(name, f1[0],f1[1]))
        f1_boxplot0.append(f1[0])
        f1_boxplot1.append(f1[1])
        f1_boxplot_av.append((f1[0]+f1[1])/2)

        fpr=FPR(TP,TN,FP,FN)
        fpr0.append(fpr[0])
        fpr1.append(fpr[1])
        fpr_av.append((fpr[0]+fpr[1])/2)

        fnr = FNR(TP,TN,FP,FN)
        fnr0.append(fnr[0])
        fnr1.append(fnr[1])
        fnr_av.append((fnr[0]+fnr[1])/2)

        precision = Precision(TP,TN,FP,FN)
        presicion0.append(precision[0])
        presicion1.append(precision[1])
        presicion_av.append((precision[0]+precision[1])/2)

        recall = Recall(TP,TN,FP,FN)
        recall0.append(recall[0])
        recall1.append(recall[1])
        recall_av.append((recall[0]+recall[1])/2)

        _zz_img1=np.round(_zz_img)
        segmentation = np.asarray(_zz_img1)
        predicted_label = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        predicted_label.SetDirection(direction=direction)
        predicted_label.SetOrigin(origin=origin)
        predicted_label.SetSpacing(spacing=voxel_size)
        sitk.WriteImage(predicted_label, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'_result.mha')

        segmentation = np.asarray(_zz_img)
        predicted_label = sitk.GetImageFromArray(segmentation.astype(np.float32))
        predicted_label.SetDirection(direction=direction)
        predicted_label.SetOrigin(origin=origin)
        predicted_label.SetSpacing(spacing=voxel_size)
        sitk.WriteImage(predicted_label,
                        '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/' + Log + out_dir + name + '_fuzzy.mha')





        segmentation = np.asarray(_zz_img_gt)
        predicted_label = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        predicted_label.SetDirection(direction=direction)
        predicted_label.SetOrigin(origin=origin)
        predicted_label.SetSpacing(spacing=voxel_size)
        sitk.WriteImage(predicted_label, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'_gtv.mha')

        segmentation = np.asarray(_zz_img_ct)
        predicted_label = sitk.GetImageFromArray(segmentation.astype(np.short))
        predicted_label.SetDirection(direction=direction)
        predicted_label.SetOrigin(origin=origin)
        predicted_label.SetSpacing(spacing=voxel_size)
        sitk.WriteImage(predicted_label, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'_ct.mha')
        # output(filename, sheet, list1, list2, x, y, z)
        # output('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir +'file.xls','sheet 1',fnr,fpr,'a','b','c')
        print('end')

    f1_bp0 = []
    f1_bp1 = []
    f1_bp_av = []
    f1_bp0.append((f1_boxplot0))
    f1_bp1.append((f1_boxplot1))
    f1_bp_av.append((f1_boxplot_av))
    plt.figure()
    plt.boxplot(f1_bp0, 0, '')
    plt.title('Tumor Dice value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'f1_bp_tumor.png')

    plt.figure()
    plt.boxplot(f1_bp1, 0, '')
    plt.title('Background Dice value for all the images '+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'f1_bp_background.png')

    plt.figure()
    plt.boxplot(f1_bp_av, 0, '')
    plt.title('Average Dice value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'f1_bp_average.png')
    #----------------------
    fpr_bp0 = []
    fpr_bp0.append((fpr0))
    plt.figure()
    plt.boxplot(fpr_bp0, 0, '')
    plt.title('FPR Tumor value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fpr_bp_tumor.png')

    fpr_bp1 = []
    fpr_bp1.append((fpr1))
    plt.figure()
    plt.boxplot(fpr_bp1, 0, '')
    plt.title('FPR Background value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fpr_bp_background.png')


    fpr_bp = []
    fpr_bp.append((fpr_av))
    plt.figure()
    plt.boxplot(fpr_bp, 0, '')
    plt.title('FPR Average value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fpr_bp_average.png')

    #----------------------
    fnr_bp0 = []
    fnr_bp0.append((fnr0))
    plt.figure()
    plt.boxplot(fnr_bp0, 0, '')
    plt.title('FNR Tumor value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fnr_bp_tumor.png')

    fnr_bp1 = []
    fnr_bp1.append((fnr1))
    plt.figure()
    plt.boxplot(fnr_bp1, 0, '')
    plt.title('FNR Background value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fnr_bp_background.png')


    fnr_bp = []
    fnr_bp.append((fnr_av))
    plt.figure()
    plt.boxplot(fnr_bp, 0, '')
    plt.title('FNR Average value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fnr_bp_average.png')
    #----------------------
    pres_bp0 = []
    pres_bp0.append((presicion0))
    plt.figure()
    plt.boxplot(pres_bp0, 0, '')
    plt.title('Precision value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'precision_bp_tumor.png')

    pres_bp1 = []
    pres_bp1.append((presicion1))
    plt.figure()
    plt.boxplot(pres_bp1, 0, '')
    plt.title('Precision Background value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'precision_bp_background.png')


    pres_bp = []
    pres_bp.append((presicion_av))
    plt.figure()
    plt.boxplot(pres_bp, 0, '')
    plt.title('Precision Average value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'precision_bp_average.png')
    #----------------------
    recall_bp0 = []
    recall_bp0.append((recall0))
    plt.figure()
    plt.boxplot(recall_bp0, 0, '')
    plt.title('Recall value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'recall_bp_tumor.png')

    recall_bp1 = []
    recall_bp1.append((recall1))
    plt.figure()
    plt.boxplot(recall_bp1, 0, '')
    plt.title('Recall Background value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'recall_bp_background.png')


    recall_bp = []
    recall_bp.append((recall_av))
    plt.figure()
    plt.boxplot(recall_bp, 0, '')
    plt.title('Recall Average value for all the images'+plot_tag)
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'recall_bp_average.png')
    #----------------------
    plt.figure()
    d_bp = []
    d_bp.append((f1_boxplot0))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, f1_boxplot0, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.45)
    plt.title('Dice all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'dice_bar.png')

    #----------------------
    plt.figure()

    fnr_bar0 = []
    fnr_bar0.append((fnr0))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, fnr0, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('FNR Background all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fnr_background_bar.png')


    #----------------------
    plt.figure()

    fnr_bar1 = []
    fnr_bar1.append((fnr1))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, fnr1, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('FNR Tumor all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fnr_tumor_bar.png')


    #----------------------
    plt.figure()

    fpr_bar0 = []
    fpr_bar0.append((fpr0))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, fpr0, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('FPR Background all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fpr_background_bar.png')


    #----------------------
    plt.figure()

    fpr_bar1 = []
    fpr_bar1.append((fpr1))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, fpr1, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('FPR tumor all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'fpr_tumor_bar.png')


    #----------------------
    plt.figure()

    recall_bar0 = []
    recall_bar0.append((recall0))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, recall0, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Recall Background all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'recall_background_bar.png')


    #----------------------
    plt.figure()

    recall_bar = []
    recall_bar.append((recall1))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, recall1, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Recall tumor all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'recall_tumor_bar.png')

    #----------------------
    plt.figure()

    recall_bar = []
    recall_bar.append((recall1))
    xs = [i for i,_ in enumerate(name_list)]

    plt.bar(xs, recall1, align='center')
    plt.xticks(xs, name_list, rotation='vertical')
    plt.margins(.05)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Recall Average all images'+plot_tag)
    plt.grid()
    plt.savefig('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/'+Log+out_dir+name+'recall_average_bar.png')


if __name__ == "__main__":
    # '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log-28-1-2018/34443_1_4_DenseUnet_esoph_lessFM_04/'

    # log_tag='/13331_0.75_4-cross-noRand-2-001/'




    Log = 'Log_2018-08-15/00_Seperate_training_2nddataset/'


    for i in range(6,7):
        fold = -i

        for j in range(0,2):
            log_tag = '13331_0.75_4-cross-noRand-train2test2--'+str(i)+'/'
            test_vali = j
            if test_vali == 1:
                out_dir = log_tag + '/result_vali/'
            else:
                out_dir = log_tag + '/result/'
            test_all_nets(fold, out_dir, Log)