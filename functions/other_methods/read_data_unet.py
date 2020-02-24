import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import matplotlib.pyplot as plt

class _read_data:
    def __init__(self):
        path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/'
        self.train_image_path=path+'prostate_train/'
        self.validation_image_path=path+'prostate_validation/'
        self.test_image_path=path+'prostate_test/'

    # ========================
    def read_imape_path(self,image_path):
        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        CTs = []
        GTVs = []
        Len = []
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]
            for dt in date:
                CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                           f.endswith('CTImage.mha')]
                GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                            f.endswith('Bladder.mha')]
                lentxt = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                          f.endswith('len.txt')]
                CTs.append(CT_path)
                GTVs.append(GTV_path)
                Len.append(lentxt)
        return CTs, GTVs, Len

    def read_image_path(self):
        '''read the name of all the images and annotaed image'''
        train_CTs, train_GTVs, train_Len=self.read_imape_path(self.train_image_path)

        validation_CTs, validation_GTVs, validation_Len=self.read_imape_path(self.validation_image_path)

        test_CTs, test_GTVs, test_Len=self.read_imape_path(self.test_image_path)

        return train_CTs, train_GTVs, train_Len,validation_CTs, validation_GTVs, \
               validation_Len,test_CTs, test_GTVs, test_Len




    def read_all_validation_batches(self,CTs,GTVs,Len,sample_no,patch_window,img_width,img_height,epoch,GTV_patchs_size,whole_image=0):



        ii = np.random.randint(0, len(CTs), size=1)
        CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
        with open(''.join(Len[int(ii)])) as f:
            lines = f.readlines()
            tumor_begin = int(lines[0])
            tumor_end = int(lines[1])

        CT_image = sitk.GetArrayFromImage(CT_image1)
        CT_image = CT_image / CT_image.mean()
        GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
        GTV_image = sitk.GetArrayFromImage(GTV_image)
        GTV_max = GTV_image.max()

        if whole_image:#here
            rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=1)
            img = CT_image[rand_depth, 0:511, 0:511]
            img1 = np.zeros((1, 595, 595))
            # img1[0][:][:] = np.lib.pad(img[0], (44, 44), "constant", constant_values=(0, 0))
            fill_val=np.max(img[0])
            img1[0][:][:] = np.lib.pad(img[0], (42, 42), "constant", constant_values=(fill_val, fill_val))
            # xlable=(np.where(img[0]==fill_val)[0])[0]
            # ylable=(np.where(img[0]==fill_val)[1])[0]
            # fill_lable=GTV_image[0][xlable][ylable]
            img=img1[..., np.newaxis]
            # imgplot = plt.imshow(img[0][:][:].reshape(600, 600), cmap='gray')


            seg1 = (GTV_image[rand_depth, 1:508,1:508])
            # seg1 = np.lib.pad(seg1, (44, 44), "constant", constant_values=(fill_lable, fill_lable))
            seg = np.eye(2)[seg1]
            return img,seg
        else:
            print("Reading %d validation batches... " % (sample_no))

            '''random numbers for selecting random samples'''
            rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=sample_no)
            rand_width = np.random.randint(int(patch_window / 2)+1, img_width - int(patch_window / 2)  ,
                                           size=int(sample_no / 2))
            rand_height = np.random.randint(int(patch_window / 2)+1, img_height - int(patch_window / 2)  ,
                                            size=int(sample_no / 2))
            '''balencing the classes:'''
            while len(rand_height) < sample_no:
                dpth = np.random.randint(0, len(rand_depth), size=1)
                ones = np.where(GTV_image[rand_depth[dpth], 0:img_width, 0:img_height] != 0)
                if len(ones[0]):
                    rnd_ones = np.random.randint(0, len(ones[0]), size=int(len(ones[0]) / 2))
                    rand_width = np.hstack((rand_width, ones[1][rnd_ones]))
                    rand_height = np.hstack((rand_height, ones[2][rnd_ones]))


            '''remove the further samples'''
            if len(rand_height) > sample_no:
                rand_width = np.delete(rand_width, list(range(sample_no, len(rand_height))), 0)
                rand_height = np.delete(rand_height, list(range(sample_no, len(rand_height))), 0)

            # with tf.device('/cpu:0'):
            validation_CT_image_patchs = np.stack([(CT_image[rand_depth[sn],
                                                    int(rand_width[sn]) - int(patch_window / 2) - 1:
                                                    int(rand_width[sn]) + int(patch_window / 2) ,
                                                    int(rand_height[sn]) - int(patch_window / 2) - 1:
                                                    int(rand_height[sn]) + int(patch_window / 2) ][..., np.newaxis]
                                                    ) for sn in
                                                   range(len(rand_height))])

            validation_GTV_image_patchs = np.stack([(GTV_image[int(rand_depth[sn]),
                         int(rand_width[sn])-int(GTV_patchs_size / 2) - 1:
                         int(rand_width[sn])+int(GTV_patchs_size / 2),
                         int(rand_height[sn])-int(GTV_patchs_size / 2) - 1:
                         int(rand_height[sn])+int(GTV_patchs_size / 2)] / GTV_max).astype(int)
                 for sn in
                 range(len(rand_height))]).reshape(sample_no, GTV_patchs_size,
                                                                                GTV_patchs_size)

            validation_GTV_label = np.eye(2)[validation_GTV_image_patchs]
            '''shuffle the lists'''

            index_shuf = list(range(len(validation_GTV_label)))
            validation_CT_image_patchs1 = np.vstack([validation_CT_image_patchs[sn][:][:]]
                                   for sn in index_shuf)
            validation_GTV_label1 = np.vstack([validation_GTV_label[sn][:][:]]
                                            for sn in index_shuf)

            return validation_CT_image_patchs1,validation_GTV_label1
#=================================================================

    def read_data_all_train_batches(self,CTs,GTVs,Len,total_sample_no,GTV_patchs_size, patch_window,img_width,img_height,epoch):
        print("Reading %d training batches... " %(total_sample_no))

        len_CT=len(CTs) # patients number
        sample_no = int(total_sample_no / len_CT) #no of samples that must be selected from each patient
        if sample_no*len_CT<total_sample_no:            #if division causes to reduce total samples
            remain=total_sample_no-sample_no*len_CT
            sample_no=sample_no+remain

        CT_image_patchs=[]
        GTV_patchs=[]
        for ii in range(len_CT): #select samples from each patient:
            # print('patient: %s' %(''.join(GTVs[int(ii)])))
            with open(''.join(Len[int(ii)])) as f: #read start and end of tumor from txt files
                lines = f.readlines()
                tumor_begin = int(lines[0])
                tumor_end = int(lines[1])

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = CT_image / CT_image.mean()


            '''random numbers for selecting random samples'''
            rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=int(sample_no/2)) #get half depth samples from the tumor region
            rand_width = np.random.randint(int(patch_window / 2)+1, img_width - int(patch_window / 2)  ,
                                           size=int(sample_no / 2)) # half of width samples
            rand_height = np.random.randint(int(patch_window / 2)+1, img_height - int(patch_window / 2)  ,
                                            size=int(sample_no / 2)) #half of width samples
            # print('0')

            '''balencing the classes:'''
            counter=0
            rand_depth1 = rand_depth  # depth sequence
            rand_width1 = rand_width  # width sequence
            rand_height1 = rand_height  # heigh sequence
            while counter < int(sample_no/2): #select half of samples from tumor only!
                # print("counter: %d" %(counter))
                dpth = np.random.randint(tumor_begin, tumor_end + 1, size=1) #select one slice
                ones = np.where(GTV_image[dpth, 0:img_width, 0:img_height] != 0) # GTV indices of slice # which belong to tumor
                if len(ones[0]): #if not empty
                    tmp=int((sample_no * .5) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]), size=tmp) #number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]), size=1)  # number of samples from each slice

                    counter += len(rnd_ones) #counter for total samples
                    rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                    rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))
                    rand_depth1= np.hstack((rand_depth1, dpth* np.ones(len(rnd_ones))))

            # print('1')
            GTV_max=GTV_image.max()

            CT_image_patchs1 =  np.stack([(CT_image[int(rand_depth1[sn]),
                                         int(rand_width1[sn]) - int(patch_window / 2)-1: int(rand_width1[
                                                                                               sn]) + int(
                                             patch_window / 2) ,
                                         int(rand_height1[sn] )- int(patch_window / 2)-1: int(rand_height1[sn]) +
                                                                                        int(patch_window / 2) ])[..., np.newaxis]
                                        for sn in  range(len(rand_height1))])
            # print('2')

            GTV_patchs1 = np.stack([(GTV_image[
                                            int(rand_depth1[sn]),
                                            int(rand_width1[sn])- int(GTV_patchs_size / 2) - 1:
                                        int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                        , int(rand_height1[sn])- int(GTV_patchs_size / 2) - 1:
                                        int(rand_height1[sn]) +int(GTV_patchs_size / 2)
                                        ]/ GTV_max).astype(int)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_height1), GTV_patchs_size,GTV_patchs_size)
            # GTV_patchs1=int(GTV_patchs1/GTV_image.max())

            # print('3')

            # print("GTV_patchs min: %d, max: %d  tumor sample no: %d" % (GTV_patchs1.min(),
            #                                                             GTV_patchs1.max(),
            #                                                             len(np.where(GTV_patchs1 != 0)[0])
            #                                                             ))
            GTV_patchs1 = np.eye(2)[GTV_patchs1]
            # print('4')

            if len(CT_image_patchs):
                CT_image_patchs=np.vstack((CT_image_patchs,CT_image_patchs1))
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs1))
            else:
                CT_image_patchs=CT_image_patchs1
                GTV_patchs = GTV_patchs1

            # print('5')

            # print("length: %d" %(len(CT_image_patchs)))


        '''remove the further samples'''
        if len(CT_image_patchs) > total_sample_no:
            CT_image_patchs = np.delete(CT_image_patchs, list(range(total_sample_no, len(CT_image_patchs))), 0)
            GTV_patchs = np.delete(GTV_patchs, list(range(total_sample_no, len(GTV_patchs))), 0)


        '''shuffle the lists'''
        index_shuf = list(range(len(CT_image_patchs)))
        shuffle(index_shuf)
        GTV_patchs1=np.vstack([GTV_patchs[sn][:][:]]
                 for sn in index_shuf)
        CT_image_patchs1=np.vstack([CT_image_patchs[sn][:][:]]
                 for sn in index_shuf)

        return GTV_patchs1,CT_image_patchs1