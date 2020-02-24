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
        path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/'
        self.train_image_path=path+'prostate_train/'
        self.validation_image_path=path+'prostate_validation/'
        self.test_image_path=path+'prostate_test/'
        self.prostate_ext_gt='Contours_Cleaned/'
        self.img_name='.mha'
        self.label_name='Bladder.mha'
        self.startwith_4DCT='4DCT_'
        self.startwith_GTV='GTV_'
        self.prostate_ext_gt = 'Contours_Cleaned/'
        self.prostate_ext_img = 'Images/'
        # self.img_name = 'CTImage.mha'
        # self.label_name = 'GTV.mha'


    # ========================
    def read_imape_path(self,image_path):
        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        CTs = []
        GTVs = []
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]
            for dt in date:
                CT_path = [(join(image_path, pd, dt, self.prostate_ext_img, f)) for f in
                           listdir(join(image_path, pd, dt, self.prostate_ext_img)) if
                           f.endswith(self.img_name)]
                GTV_path = [join(image_path, pd, dt, self.prostate_ext_gt, f) for f in
                            listdir(join(image_path, pd, dt, self.prostate_ext_gt)) if
                            f.endswith(self.label_name)]

                CTs.append(CT_path)
                GTVs.append(GTV_path)

        return CTs, GTVs


    def read_image_path(self):
        '''read the name of all the images and annotaed image'''
        train_CTs, train_GTVs=self.read_imape_path(self.train_image_path)

        validation_CTs, validation_GTVs=self.read_imape_path(self.validation_image_path)

        test_CTs, test_GTVs=self.read_imape_path(self.test_image_path)

        return train_CTs, train_GTVs,validation_CTs, validation_GTVs, \
               test_CTs, test_GTVs




    # def read_all_validation_batches1(self,CTs,GTVs,sample_no,patch_window,img_width,img_height,epoch,GTV_patchs_size,whole_image=0):
    #
    #
    #     ii = np.random.randint(0, len(CTs), size=1)
    #     CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
    #
    #
    #     CT_image = sitk.GetArrayFromImage(CT_image1)
    #     CT_image = (CT_image - CT_image.mean())/CT_image.std()
    #
    #
    #     GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
    #     GTV_image = sitk.GetArrayFromImage(GTV_image)
    #     GTV_max = GTV_image.max()
    #     tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
    #     tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])
    #
    #     if whole_image:
    #         rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=1)
    #
    #         img = CT_image[rand_depth, 0:img_height, 0:img_height]
    #         img_padded_size=527
    #         img1 = np.zeros((1, img_padded_size, img_padded_size))
    #         fill_val = img[0][0][0]
    #         img1[0][:][:] = np.lib.pad(img[0], (int((img_padded_size-img_height)/2), int((img_padded_size-img_height)/2)),
    #                                    "constant", constant_values=(fill_val, fill_val))
    #         img = img1[..., np.newaxis]
    #
    #         seg_size=497
    #
    #         seg1 = (GTV_image[rand_depth, img_height-int(seg_size/2)-1:img_height+int(seg_size/2),
    #                 img_height - int(seg_size / 2) - 1:img_height + int(seg_size / 2)])
    #         seg = np.eye(2)[seg1]
    #         return img,seg
    #     else:
    #         print("Reading %d validation batches... " % (sample_no))
    #
    #         '''random numbers for selecting random samples'''
    #
    #         rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=sample_no)
    #
    #
    #         rand_width = np.random.randint(int(patch_window / 2)+1, img_width - int(patch_window / 2) ,
    #                                        size=int(sample_no / 2))
    #         rand_height = np.random.randint(int(patch_window / 2)+1, img_height - int(patch_window / 2) ,
    #                                         size=int(sample_no / 2))
    #         '''balencing the classes:'''
    #         while len(rand_height) < sample_no:
    #             dpth = np.random.randint(0, len(rand_depth), size=1)
    #             ones = np.where(GTV_image[rand_depth[dpth], 0:img_width, 0:img_height] != 0)
    #             if len(ones[0]):
    #                 rnd_ones = np.random.randint(0, len(ones[0]), size=int(len(ones[0]) / 2))
    #                 rand_width = np.hstack((rand_width, ones[1][rnd_ones]))
    #                 rand_height = np.hstack((rand_height, ones[2][rnd_ones]))
    #
    #
    #         '''remove the further samples'''
    #         if len(rand_height) > sample_no:
    #             rand_width = np.delete(rand_width, list(range(sample_no, len(rand_height))), 0)
    #             rand_height = np.delete(rand_height, list(range(sample_no, len(rand_height))), 0)
    #
    #         # with tf.device('/cpu:0'):
    #         validation_CT_image_patchs = np.stack([(CT_image[rand_depth[sn],
    #                                                 int(rand_width[sn]) - int(patch_window / 2) - 1:
    #                                                 int(rand_width[sn]) + int(patch_window / 2) ,
    #                                                 int(rand_height[sn]) - int(patch_window / 2) - 1:
    #                                                 int(rand_height[sn]) + int(patch_window / 2) ][..., np.newaxis]
    #                                                 ) for sn in
    #                                                range(len(rand_height))])
    #
    #         validation_GTV_image_patchs = np.stack([(GTV_image[int(rand_depth[sn]),
    #                      int(rand_width[sn])-int(GTV_patchs_size / 2)-1:
    #                      int(rand_width[sn])+int(GTV_patchs_size / 2),
    #                      int(rand_height[sn])-int(GTV_patchs_size / 2)-1:
    #                      int(rand_height[sn])+int(GTV_patchs_size / 2)] / GTV_max).astype(int)
    #              for sn in
    #              range(len(rand_height))]).reshape(sample_no, GTV_patchs_size,
    #                                                                             GTV_patchs_size)
    #
    #         # print("length: %d" % (len(validation_GTV_image_patchs)))
    #
    #         validation_GTV_label = np.eye(2)[validation_GTV_image_patchs]
    #         '''shuffle the lists'''
    #
    #         index_shuf = list(range(len(validation_GTV_label)))
    #         validation_CT_image_patchs1 = np.vstack([validation_CT_image_patchs[sn][:][:]]
    #                                for sn in index_shuf)
    #         validation_GTV_label1 = np.vstack([validation_GTV_label[sn][:][:]]
    #                                         for sn in index_shuf)
    #
    #         return validation_CT_image_patchs1,validation_GTV_label1

# =================================================================

    def read_image_seg_volume(self, CTs, GTVs,img_index):
        # ii = np.random.randint(0, len(CTs), size=1)
        CT_image1 = sitk.ReadImage(''.join(CTs[int(img_index)]))
        voxel_size=CT_image1.GetSpacing()
        origin=CT_image1.GetOrigin()
        direction=CT_image1.GetDirection()

        CT_image = sitk.GetArrayFromImage(CT_image1)
        CT_image = (CT_image)/CT_image.mean()

        GTV_image = sitk.ReadImage(''.join(GTVs[int(img_index)]))
        GTV_image = sitk.GetArrayFromImage(GTV_image)
        return CT_image,GTV_image,GTV_image.shape[0],voxel_size,origin,direction
# =================================================================
    def read_image(self, CT_image,GTV_image, img_height, img_padded_size,seg_size,depth):
        img = CT_image[depth, 0:img_height-1, 0:img_height-1]
        img1 = np.zeros((1, img_padded_size, img_padded_size))
        fill_val = img[0][0]
        img1[0][:][:] = np.lib.pad(img, (
        int((img_padded_size - img_height) / 2+1), int((img_padded_size - img_height) / 2+1)),
                                   "constant", constant_values=(fill_val, fill_val))
        img = img1[..., np.newaxis]
        seg1 = (GTV_image[depth, int(img_height/2) - int(seg_size / 2) - 1:int(img_height/2) + int(seg_size / 2),
                int(img_height / 2) - int(seg_size / 2) - 1:int(img_height/2) + int(seg_size / 2)])
        seg = np.eye(2)[seg1]
        seg=seg[np.newaxis]
        return img,seg
# =================================================================
    def check(self,GTVs,width_patch,height_patch,depth_patch):
        no_of_images = len(GTVs)
        for ii in range(no_of_images):
            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            if(max(depth_patch[ii])>len(GTV_image)):
                print('error')

# =================================================================
    def read_patche_indice_online(self,CTs, GTVs, total_sample_no,patch_window,img_width,img_height):
        no_of_images=len(CTs)
        patch_no_per_image=int(total_sample_no/no_of_images)
        if patch_no_per_image==0:
            patch_no_per_image=1
        width_patch=[]
        height_patch=[]
        depth_patch=[]
        rand_depth20=[]
        rand_width20=[]
        rand_height20=[]
        rand_depth21 = []
        rand_width21 = []
        rand_height21 = []
        for ii in range(no_of_images):
            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

            '''random numbers for selecting random samples'''
            rand_depth = np.random.randint(int(patch_window / 2) + 1, len(GTV_image) - int(patch_window / 2)  ,
                                           size=int(patch_no_per_image / 2))  # get half depth samples from every where
            rand_width = np.random.randint(int(patch_window / 2) + 1, img_width - int(patch_window / 2),
                                           size=int(patch_no_per_image / 2))  # half of width samples
            rand_height = np.random.randint(int(patch_window / 2) + 1, img_height - int(patch_window / 2),
                                            size=int(patch_no_per_image / 2))  # half of width samples

            # '''shuffle the lists'''
            # index_shuf = list(range(len(rand_width)))
            # shuffle(index_shuf)
            # rand_width22 = np.hstack([rand_width[sn]]
            #                          for sn in index_shuf)
            # rand_depth22 = np.hstack([rand_depth[sn]]
            #                          for sn in index_shuf)
            # rand_height22 = np.hstack([rand_height[sn]]
            #                           for sn in index_shuf)

            # [rand_width00,rand_height00,rand_depth00]=self.shuffle_lists(rand_width,rand_height,rand_depth)

            '''balencing the classes:'''
            counter = 0
            rand_depth1 = []  # depth sequence
            rand_width1 = []  # width sequence
            rand_height1 = []  # heigh sequence
            while counter < int(patch_no_per_image / 2):  # select half of samples from tumor only!
                # print("counter: %d" %(counter))
                if tumor_begin-int(patch_window/2)<0:
                    begin=int(patch_window / 2) + 1
                else:
                    begin = tumor_begin
                if tumor_end+int(patch_window/2)>=len(GTV_image):
                   end= len(GTV_image) - int(patch_window / 2)-1
                else:
                   end=tumor_end

                dpth = np.random.randint(begin, end, size=1)  # select one slice
                ones = np.where(GTV_image[dpth, 0:img_width,
                                0:img_height] != 0)  # GTV indices of slice which belong to tumor


                if len(ones[0]):  # if not empty
                    tmp = int((patch_no_per_image * .5) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=tmp)  # number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]),
                                                     size=1)  # number of samples from each slice

                    counter += len(rnd_ones)  # counter for total samples
                    rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                    rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))
                    rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))



            # '''shuffle the lists'''
            # index_shuf = list(range(len(rand_width1)))
            # shuffle(index_shuf)
            # rand_width11 = np.hstack([rand_width1[sn]]
            #                          for sn in index_shuf)
            # rand_depth11 = np.hstack([rand_depth1[sn]]
            #                          for sn in index_shuf)
            # rand_height11 = np.hstack([rand_height1[sn]]
            #                           for sn in index_shuf)
            # [rand_width11, rand_height11, rand_depth11] = self.shuffle_lists(rand_width1, rand_height1, rand_depth1)



            rand_depth20.append(rand_depth)
            rand_width20.append(rand_width)
            rand_height20.append(rand_height)

            rand_depth21.append(rand_depth1)
            rand_width21.append(rand_width1)
            rand_height21.append(rand_height1)


        for i in range(len(rand_depth20)):
            [w0,h0,d0] = self.shuffle_lists(rand_width20[i], rand_height20[i], rand_depth20[i])
            [w1,h1,d1] = self.shuffle_lists(rand_width21[i], rand_height21[i], rand_depth21[i])
            depth_patch.append(np.hstack((d0, d1)))
            width_patch.append(np.hstack((w0, w1)))
            height_patch.append(np.hstack((h0, h1)))







        if len(width_patch)==0:
            print('----> %d , %d'%(len(width_patch),len(height_patch)))
        return width_patch,height_patch,depth_patch
        # =================================================================
    def shuffle_lists(self,rand_width1,rand_height1,rand_depth1):
        index_shuf = list(range(len(rand_width1)))
        shuffle(index_shuf)
        rand_width11 = np.hstack([rand_width1[sn]]
                                 for sn in index_shuf)
        rand_depth11 = np.hstack([rand_depth1[sn]]
                                 for sn in index_shuf)
        rand_height11 = np.hstack([rand_height1[sn]]
                                  for sn in index_shuf)
        return rand_width11,rand_height11,rand_depth11

# =================================================================

    def read_all_validation_batches(self, CTs, GTVs, total_sample_no, GTV_patchs_size, patch_window, img_width,
                                    img_height, epoch,img_padded_size,seg_size,whole_image=0):
        if whole_image:
            # img_padded_size = 519
            # seg_size = 505

            ii = np.random.randint(0, len(CTs), size=1)
            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))

            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = (CT_image)/CT_image.mean()

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            # GTV_max = GTV_image.max()

            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

            # rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=1)
            # img = CT_image[rand_depth, 0:511, 0:511]
            # img1 = np.zeros((1, 1023, 1023))
            # fill_val = img[0][0][0]
            # img1[0][:][:] = np.lib.pad(img[0], (256, 256), "constant", constant_values=(fill_val, fill_val))
            # img = img1[..., np.newaxis]
            #
            # seg1 = (GTV_image[rand_depth, 6:503, 6:503])
            # seg = np.eye(2)[seg1]

            rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=1)

            img = CT_image[rand_depth, 0:img_height-1, 0:img_height-1]


            img1 = np.zeros((1, img_padded_size, img_padded_size))
            fill_val = img[0][0][0]
            img1[0][:][:] = np.lib.pad(img[0], (
            int((img_padded_size - img_height) / 2+1), int((img_padded_size - img_height) / 2+1)),
                                       "constant", constant_values=(fill_val, fill_val))
            img = img1[..., np.newaxis]



            seg1 = (GTV_image[rand_depth, int(img_height/2) - int(seg_size / 2) - 1:int(img_height/2) + int(seg_size / 2),
                    int(img_height / 2) - int(seg_size / 2) - 1:int(img_height/2) + int(seg_size / 2)])
            seg = np.eye(2)[seg1]
            return img,seg


        else:
            print("Reading %d Validation batches... " % (total_sample_no))

            len_CT = len(CTs)  # patients number
            sample_no = int(total_sample_no / len_CT)  # no of samples that must be selected from each patient
            if sample_no * len_CT < total_sample_no:  # if division causes to reduce total samples
                remain = total_sample_no - sample_no * len_CT
                sample_no = sample_no + remain

            CT_image_patchs = []
            GTV_patchs = []
            for ii in range(len_CT):  # select samples from each patient:

                GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
                GTV_image = sitk.GetArrayFromImage(GTV_image)
                tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
                tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

                CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
                CT_image = sitk.GetArrayFromImage(CT_image1)
                CT_image = (CT_image )/CT_image.mean()
                '''random numbers for selecting random samples'''
                rand_depth = np.random.randint(0, len(GTV_image),
                                               size=int(sample_no / 2))  # get half depth samples from every where
                rand_width = np.random.randint(int(patch_window / 2) + 1, img_width - int(patch_window / 2),
                                               size=int(sample_no / 2))  # half of width samples
                rand_height = np.random.randint(int(patch_window / 2) + 1, img_height - int(patch_window / 2),
                                                size=int(sample_no / 2))  # half of width samples
                # print('0')

                '''balencing the classes:'''
                counter = 0
                rand_depth1 = rand_depth  # depth sequence
                rand_width1 = rand_width  # width sequence
                rand_height1 = rand_height  # heigh sequence
                while counter < int(sample_no / 2):  # select half of samples from tumor only!
                    # print("counter: %d" %(counter))
                    dpth = np.random.randint(tumor_begin, tumor_end + 1, size=1)  # select one slice
                    ones = np.where(GTV_image[dpth, 0:img_width,
                                    0:img_height] != 0)  # GTV indices of slice which belong to tumor
                    if len(ones[0]):  # if not empty
                        tmp = int((sample_no * .5) / (tumor_end - tumor_begin))
                        if tmp:
                            rnd_ones = np.random.randint(0, len(ones[0]),
                                                         size=tmp)  # number of samples from each slice
                        else:
                            rnd_ones = np.random.randint(0, len(ones[0]),
                                                         size=1)  # number of samples from each slice

                        counter += len(rnd_ones)  # counter for total samples
                        rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                        rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))
                        rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))

                # print('1')
                GTV_max = GTV_image.max()

                CT_image_patchs1 = np.stack([(CT_image[int(rand_depth1[sn]),
                                              int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[
                                                                                                        sn]) + int(
                                                  patch_window / 2),
                                              int(rand_height1[sn]) - int(patch_window / 2) - 1: int(
                                                  rand_height1[sn]) +
                                                                                                 int(
                                                                                                     patch_window / 2)])[
                                                 ..., np.newaxis]
                                             for sn in range(len(rand_height1))])
                # print('2')

                GTV_patchs1 = np.stack([(GTV_image[
                                         int(rand_depth1[sn]),
                                         int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                         int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                         , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                         int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                         ] / GTV_max).astype(int)
                                        for sn in
                                        range(len(rand_height1))]).reshape(len(rand_height1), GTV_patchs_size,
                                                                           GTV_patchs_size)
                # GTV_patchs1=int(GTV_patchs1/GTV_image.max())

                # print('3')

                # print("GTV_patchs min: %d, max: %d  tumor sample no: %d" % (GTV_patchs1.min(),
                #                                                             GTV_patchs1.max(),
                #                                                             len(np.where(GTV_patchs1 != 0)[0])
                #                                                             ))
                GTV_patchs1 = np.eye(2)[GTV_patchs1]
                # print('4')

                if len(CT_image_patchs):
                    CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs1))
                    GTV_patchs = np.vstack((GTV_patchs, GTV_patchs1))
                else:
                    CT_image_patchs = CT_image_patchs1
                    GTV_patchs = GTV_patchs1

                # print('5')

                # print("length: %d" % (len(CT_image_patchs)))

            '''remove the further samples'''
            if len(CT_image_patchs) > total_sample_no:
                CT_image_patchs = np.delete(CT_image_patchs, list(range(total_sample_no, len(CT_image_patchs))), 0)
                GTV_patchs = np.delete(GTV_patchs, list(range(total_sample_no, len(GTV_patchs))), 0)

            '''shuffle the lists'''
            index_shuf = list(range(len(CT_image_patchs)))
            shuffle(index_shuf)
            GTV_patchs1 = np.vstack([GTV_patchs[sn][:][:]]
                                    for sn in index_shuf)
            CT_image_patchs1 = np.vstack([CT_image_patchs[sn][:][:]]
                                         for sn in index_shuf)

        return CT_image_patchs1,GTV_patchs1
#=================================================================


    def read_data_all_train_batches(self,CTs,GTVs,total_sample_no,GTV_patchs_size, patch_window,img_width,img_height,epoch):
        print("Reading %d training batches... " %(total_sample_no))

        len_CT=len(CTs) # patients number
        sample_no = int(total_sample_no / len_CT) #no of samples that must be selected from each patient
        if sample_no*len_CT<total_sample_no:            #if division causes to reduce total samples
            # remain=total_sample_no-sample_no*len_CT
            sample_no=sample_no+2

        CT_image_patchs=[]
        GTV_patchs=[]
        for ii in range(len_CT): #select samples from each patient:

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = (CT_image )/CT_image.mean()
            '''random numbers for selecting random samples'''
            rand_depth = np.random.randint(0,len(GTV_image), size=int(sample_no/2)) #get half depth samples from every where
            rand_width = np.random.randint(int(patch_window / 2)+1, img_width - int(patch_window / 2) ,
                                           size=int(sample_no / 2)) # half of width samples
            rand_height = np.random.randint(int(patch_window / 2)+1, img_height - int(patch_window / 2) ,
                                            size=int(sample_no / 2)) #half of width samples
            # print('0')

            '''balencing the classes:'''
            counter=0
            rand_depth1 = rand_depth  # depth sequence
            rand_width1 = rand_width  # width sequence
            rand_height1 = rand_height  # heigh sequence
            while counter < int(sample_no/2): #select half of samples from tumor only!
                # print("c
                # ounter: %d" %(counter))
                dpth = np.random.randint(tumor_begin, tumor_end + 1, size=1) #select one slice
                ones = np.where(GTV_image[dpth, 0:img_width, 0:img_height] != 0) # GTV indices of slice which belong to tumor
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
                                         int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[
                                                                                               sn]) + int(
                                             patch_window / 2) ,
                                         int(rand_height1[sn] )- int(patch_window / 2) - 1: int(rand_height1[sn]) +
                                                                                        int(patch_window / 2)])[..., np.newaxis]
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