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
    def __init__(self, data, train_tag='prostate_train/', validation_tag='prostate_validation/',
                 test_tag='prostate_test/',
                 img_name='.mha', label_name='Bladder.mha', torso_tag='Torso.mha',resampled_path=''):

        path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/'
        self.train_image_path = path + train_tag
        self.validation_image_path = path + validation_tag
        self.test_image_path = path + test_tag
        self.prostate_ext_gt = 'Contours_Cleaned/'
        self.startwith_4DCT = '4DCT_'
        self.startwith_GTV = 'GTV_'
        self.prostate_ext_gt = 'Contours_Cleaned/'
        self.prostate_ext_img = 'Images/'
        self.img_name = img_name
        self.label_name = label_name
        self.data = data
        # self.torso_tag=torso_tag
        self.patchsize = 87
        self.resample_tag = '_re113'
        self.img_name = 'CT' + self.resample_tag + 'z' + '_pad' + str(
            self.patchsize) + '.mha'  # I added z because it is already masked by torso! out of torso is equal to -1024!
        self.label_name1 = 'GTV_CT' + self.resample_tag + '_pad' + str(self.patchsize) + '.mha'
        self.label_name2 = 'GTV' + self.resample_tag + '_pad' + str(self.patchsize) + '.mha'
        self.ttag = '_Torso'
        self.torso_tag = 'CT' + self.ttag + self.resample_tag + '_pad' + str(self.patchsize) + '.mha'
        self.penalize_tag = 'CT' + self.resample_tag + '_penalize.mha'
        self.startwith_4DCT = '4DCT_'
        self.startwith_GTV = 'GTV_'

        self.resampled_path = resampled_path#'/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v3/'
        self.seed=1

    # ========================
    def read_train_data(self, data_dir):

        image_path = self.resampled_path
        CTs = []
        GTVs = []
        Torsos = []
        Penalize = []
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]

            for dt in date:
                # read CT and GTV images
                CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                           f.startswith(self.img_name)]
                GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                            f.endswith(self.label_name1)]
                Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                              f.endswith(self.torso_tag)]
                Penalize_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                 f.endswith(self.penalize_tag)]
                if (len(GTV_path) == 0):
                    GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                f.endswith(self.label_name2)]

                CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                             f.startswith(self.startwith_4DCT) & f.endswith(
                                 '%' + self.resample_tag + 'z' + '_pad' + str(self.patchsize) + '.mha')]
                CT_path = CT_path + CT4D_path  # here sahar
                for i in range(len(CT4D_path)):
                    percent = CT4D_path[i].split('/')[-1].split('.')[0].split('_')[1]
                    name_gtv4d = 'GTV_4DCT_' + percent + self.resample_tag + '_pad' + str(self.patchsize) + '.mha'
                    GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                    Torso_gtv4d = self.startwith_4DCT + percent + self.ttag + self.resample_tag + '_pad' + str(
                        self.patchsize) + '.mha'
                    Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))

                    penalize_gtv4d = self.startwith_4DCT + percent + '_penalize.mha'
                    Penalize_path.append((str(join(image_path, pd, dt, penalize_gtv4d))))

                CTs += (CT_path)
                GTVs += (GTV_path)
                Torsos += (Torso_path)
                Penalize += (Penalize_path)

        CTs = np.sort(CTs)
        GTVs = np.sort(GTVs)
        Torsos = np.sort(Torsos)
        Penalize = np.sort(Penalize)
        return CTs, GTVs, Torsos, Penalize

    # ========================
    def read_data_path(self, fold):  # join(self.resampled_path, f)
        data_dir = [join(self.resampled_path, f)
                    for f in listdir(self.resampled_path)
                    if (not (isfile(join(self.resampled_path, f))) and
                        not (os.path.isfile(join(self.resampled_path, f + '/delete.txt'))))]
        data_dir = np.sort(data_dir)

        # train_CTs,train_GTVs,train_Torso,train_penalize=self.read_train_data(np.hstack((data_dir[0:15],data_dir[21:101])))
        # validation_CTs,validation_GTVs,validation_Torso,validation_penalize=self.read_train_data(np.hstack((data_dir[18:21],data_dir[101:104])))
        # test_CTs,test_GTVs,test_Torso,test_penalize=self.read_train_data(np.hstack((data_dir[15:18],data_dir[104:130])))

        # 6th: train: 0-16, 21-91, validation: 19-21, 91-101, test: 16:19, 101:-1
        # 7th: train: 6-21, 41-111, validation: 3-6, 21-41, test: 0:3, 111:-1
        if fold<0:

            if fold==-1: #training on the 2nd dataset

                train_indx2 = []
                test_indx2 = []
                train_indx1 = list(range(31, 127))

                valid_indx1 = list(range(19, 21))
                valid_indx2 = list(range(21, 31))

                test_indx1 = list(range(0, 19))

            if fold==-2: #training on the 2nd dataset
                train_indx2 = []
                train_indx1 = list(range(41, 127))

                valid_indx1 = list(range(19, 21))
                valid_indx2 = list(range(21, 31))

                test_indx1 = list(range(0, 19))
                test_indx2=list(range(31, 41))
#============================================================================
            if fold==-3: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                train_indx1 = list(range(31, 101))

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(21, 31))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))
            if fold==-4: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                A = (list(range(31, 41)), list(range(51, 101)))
                train_indx1 = sum(A, [])

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(41, 51))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))
            if fold==-5: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                A = (list(range(21, 51)), list(range(61, 101)))
                train_indx1 = sum(A, [])

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(51, 61))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))
            if fold==-6: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                A = (list(range(21, 61)), list(range(71, 101)))
                train_indx1 = sum(A, [])

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(61, 71))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))
            if fold==-7: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                A = (list(range(21, 71)), list(range(81, 101)))
                train_indx1 = sum(A, [])

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(71, 81))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))
            if fold==-8: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                A=(list(range(21, 31)), list(range(41, 101)))
                train_indx1 = sum(A, [])

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(31, 41))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))
            if fold==-9: #training on the 2nd dataset test on 2nd dataset
                train_indx2 = []
                A = (list(range(21, 81)), list(range(91, 101)))
                train_indx1 = sum(A, [])

                valid_indx1 = []#list(range(19, 21))
                valid_indx2 = list(range(81, 91))

                test_indx1 = []#list(range(0, 19))
                test_indx2=list(range(101, 127))

# ============================================================================
        elif fold>100: #train on the 1st db
            if fold==101:
                train_indx1 = list(range(0, 13))
                train_indx2 = []
                valid_indx1 = list(range(13, 15))
                valid_indx2 = []
            elif fold==102:
                train_indx1 = list(range(2, 15))
                train_indx2 = []
                valid_indx1 = list(range(0, 2))
                valid_indx2 = []
            elif fold==103:
                A = (list(range(0, 2)), list(range(4, 15)))
                train_indx1 = sum(A, [])
                train_indx2 = []
                valid_indx1 = list(range(2, 4))
                valid_indx2 = []
            elif fold==104:
                A = (list(range(0, 4)), list(range(6, 15)))
                train_indx1 = sum(A, [])
                train_indx2 = []
                valid_indx1 = list(range(4, 6))
                valid_indx2 = []
            elif fold==105:
                A = (list(range(0, 6)), list(range(8, 15)))
                train_indx1 = sum(A, [])
                train_indx2 = []
                valid_indx1 = list(range(6, 8))
                valid_indx2 = []
            elif fold==106:
                A = (list(range(0, 8)), list(range(10, 15)))
                train_indx1 = sum(A, [])
                train_indx2 = []
                valid_indx1 = list(range(8, 10))
                valid_indx2 = []
            elif fold==107:
                A = (list(range(0, 12)), list(range(14, 15)))
                train_indx1 = sum(A, [])
                train_indx2 = []
                valid_indx1 = list(range(12, 14))
                valid_indx2 = []
            test_indx1 = list(range(15, 21))
            test_indx2 = list(range(21, 127))


        else:
            if fold==0 : #train on both datasets
                # select indices for 5-fold cross validation:
                train_indx1 = list(range(0, 13))
                train_indx2 = list(range(21, 91))
                valid_indx1 = list(range(19, 21))
                valid_indx2 = list(range(91, 101))

            # fold :1
            if fold==1:
                A = (list(range(2, 13)),list(range(19, 21)))
                train_indx1=sum(A, [])
                train_indx2 = list(range(31, 101))
                valid_indx1 = list(range(0, 2))
                valid_indx2 = list(range(21, 31))

            # fold :2
            if fold == 2:
                A = (list(range(0, 2)),list(range(4, 13)), list(range(19, 21)))
                train_indx1 = sum(A, [])
                A = (list(range(21, 31)),list(range(41, 101)))
                train_indx2 = sum(A, [])
                valid_indx1 = list(range(2, 4))
                valid_indx2 = list(range(31, 41))


            if fold==3:
                diffcult_cases = [3, 5, 11, 15, 17, 20]
                train_indx2 = []
                A=(diffcult_cases)
                train_indx1 = sum(A, [])#list(range(41, 127))

                valid_indx1 = list(range(19, 21))
                valid_indx2 = list(range(21, 31))
            if fold==5:
                A = (list(range(0, 5)), list(range(7, 13)), list(range(19, 21)))
                train_indx1 = sum(A, [])

                A = (list(range(21, 51)), list(range(61, 101)))
                train_indx2 = sum(A, [])

                valid_indx1 = list(range(5, 7))
                valid_indx2 = list(range(51, 61))

            test_indx1 = list(range(13, 19))
            test_indx2 = list(range(101, 127))

        # offset = 36
        # train_vali_indx1=list(range(0,15)) #3 for validation
        # train_vali_indx2=list(range(offset,127)) #3 for validation
        #
        # valid_indx1=list(range(fold*3,(fold+1)*3))
        # valid_indx2=list(range(offset+fold*3,offset+(fold+1)*3))
        #
        # train_indx1=[x for x in train_vali_indx1 if x not in valid_indx1]
        # train_indx2=[x for x in train_vali_indx2 if x not in valid_indx2]

        train_CTs, train_GTVs, train_Torso, train_penalize = self.read_train_data(
            np.hstack((data_dir[train_indx1], data_dir[train_indx2])))
        validation_CTs, validation_GTVs, validation_Torso, validation_penalize = self.read_train_data(
            np.hstack((data_dir[valid_indx1], data_dir[valid_indx2])))
        test_CTs, test_GTVs, test_Torso, test_penalize = self.read_train_data(
            np.hstack((data_dir[test_indx1], data_dir[test_indx2])))
        print(
            'train: %d dataset#1, %d dataset#2 \nvalidation: %d dataset#1, %d dataset#2 \ntest: %d dataset#1, %d dataset#2 \ntotal: %d' %
            (len(data_dir[train_indx1]), len(data_dir[train_indx2]), len(data_dir[valid_indx1]),
             len(data_dir[valid_indx2]),
             len(data_dir[test_indx1]), len(data_dir[test_indx2]),
             len(data_dir[train_indx1]) + len(data_dir[train_indx2]) +
             len(data_dir[valid_indx1]) + len(data_dir[valid_indx2]) +
             len(data_dir[test_indx1]) + len(data_dir[test_indx2])))
        # already read all the path, now select train, validation and test!

        return train_CTs, train_GTVs, train_Torso, train_penalize, \
               validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
               test_CTs, test_GTVs, test_Torso, test_penalize

    # ========================
    def read_imape_path3(self, image_path):  # for padding_images
        CTs = []
        GTVs = []
        Torsos = []
        resampletag = '_re113'
        img_name = 'CT' + resampletag + 'z.mha'
        label_name = 'GTV_CT' + resampletag + '.mha'
        label_name2 = 'GTV' + resampletag + '.mha'
        torso_tag = 'CT_Torso' + resampletag + '.mha'

        startwith_4DCT = '4DCT_'
        startwith_GTV = 'GTV_'

        img_name = img_name
        label_name = label_name
        torso_tag = torso_tag

        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]
            for dt in date:
                # read CT and GTV images
                CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                           f.startswith(img_name)]
                GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                            f.endswith(label_name) or f.endswith(label_name2)]
                # print GTV_path
                Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                              f.endswith(torso_tag)]

                # print('%s\n%s\n%s' % (
                # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                             f.startswith(startwith_4DCT) & f.endswith('%' + self.resample_tag + 'z.mha')]

                CT_path = CT_path + CT4D_path  # here sahar
                for i in range(len(CT4D_path)):
                    # print CT4D_path[i]
                    name_gtv4d = 'GTV_' + CT4D_path[i].split('/')[10].split('z.')[0] + '.mha'
                    # print('name:%s'%(name_gtv4d))
                    GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                    Torso_gtv4d = CT4D_path[i].split('/')[10].split(self.resample_tag + 'z.')[
                                      0] + '_Torso' + self.resample_tag + '.mha'
                    # print('***********'+Torso_gtv4d)
                    Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))
                # print (';;;;'+CT4D_path[i])

                # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                CTs += (CT_path)
                GTVs += (GTV_path)
                Torsos += (Torso_path)
        return CTs, GTVs, Torsos

    # ==========================
    def read_imape_path2(self, image_path):  # for mask_image_by_torso
        CTs = []
        GTVs = []
        Torsos = []
        resampletag = '_re113'
        img_name = 'CT' + resampletag + 'z' + '_pad' + str(self.patchsize) + '.mha'
        label_name = 'GTV_CT' + resampletag + '_pad' + str(self.patchsize) + '.mha'
        label_name2 = 'GTV' + resampletag + '_pad' + str(self.patchsize) + '.mha'
        torso_tag = 'CT_Torso' + resampletag + '_pad' + str(self.patchsize) + '.mha'

        startwith_4DCT = '4DCT_'
        startwith_GTV = 'GTV_'

        img_name = img_name
        label_name = label_name
        torso_tag = torso_tag

        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]
            for dt in date:
                # read CT and GTV images
                CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                           f.startswith(img_name)]
                GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                            f.endswith(label_name) or f.endswith(label_name2)]
                # print GTV_path
                Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                              f.endswith(torso_tag)]

                # print('%s\n%s\n%s' % (
                # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                             f.startswith(startwith_4DCT) & f.endswith(
                                 '%' + self.resample_tag + 'z_pad' + str(self.patchsize) + '.mha')]

                CT_path = CT_path + CT4D_path  # here sahar
                for i in range(len(CT4D_path)):
                    # print CT4D_path[i]
                    name_gtv4d = 'GTV_' + CT4D_path[i].split('/')[10].split('z_pad' + str(self.patchsize) + '.')[
                        0] + '_pad' + str(self.patchsize) + '.mha'
                    # print('name:%s'%(name_gtv4d))
                    GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                    Torso_gtv4d = CT4D_path[i].split('/')[10].split('_re113z_pad' + str(self.patchsize) + '.')[
                                      0] + '_Torso' + self.resample_tag + '_pad' + str(self.patchsize) + '.mha'
                    # print('***********'+Torso_gtv4d)
                    Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))
                # print (';;;;'+CT4D_path[i])

                # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                CTs += (CT_path)
                GTVs += (GTV_path)
                Torsos += (Torso_path)
        return CTs, GTVs, Torsos

    # ========================
    def read_volume(self, path):
        ct = sitk.ReadImage(path)
        voxel_size = ct.GetSpacing()
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        ct = sitk.GetArrayFromImage(ct)
        return ct, voxel_size, origin, direction

    # ========================
    def read_imape_path(self, image_path):
        CTs = []
        GTVs = []
        Torsos = []
        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        if self.data == 1:
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

            return CTs, GTVs, Torsos
        elif self.data == 2:
            for pd in data_dir:
                date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                        ~isfile(join(image_path, pd, dt))]

                for dt in date:

                    # read CT and GTV images
                    CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                               f.startswith(self.img_name)]
                    GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                f.endswith(self.label_name)]
                    Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                  f.endswith(self.torso_tag)]

                    # print('%s\n%s\n%s' % (
                    # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                    CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                                 f.startswith(self.startwith_4DCT) & f.endswith('_padded.mha')]
                    CT_path = CT_path + CT4D_path  # here sahar
                    for i in range(len(CT4D_path)):
                        name_gtv4d = 'GTV_4DCT_' + CT4D_path[i].split('/')[10].split('.')[0].split('_')[
                            1] + '_padded.mha'
                        GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                        Torso_gtv4d = CT4D_path[i].split('/')[10].split('.')[0] + '_Torso.mha'
                        Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))

                        # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                    CTs += (CT_path)
                    GTVs += (GTV_path)
                    Torsos += (Torso_path)

            return CTs, GTVs, Torsos

    def read_image_path2(self):
        '''read the name of all the images and annotaed image'''
        print('Looking for the normal constants, please wait...')
        train_CTs, train_GTVs, train_Torso = self.read_imape_path(self.train_image_path)
        [min1, max1] = self.return_normal_const(train_CTs)

        validation_CTs, validation_GTVs, validation_Torso = self.read_imape_path(self.validation_image_path)
        [min2, max2] = self.return_normal_const(validation_CTs)

        test_CTs, test_GTVs, test_Torso = self.read_imape_path(self.test_image_path)
        [min3, max3] = self.return_normal_const(test_CTs)

        min_normal = np.min([min1, min2, min3])
        max_normal = np.max([max1, max2, max3])
        # [depth,width,height]=self.return_depth_width_height( CTs)
        return train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
               test_CTs, test_GTVs, test_Torso, min_normal, max_normal

    def read_image_path(self):
        '''read the name of all the images and annotaed image'''
        train_CTs, train_GTVs, train_Torso = self.read_imape_path(self.train_image_path)

        validation_CTs, validation_GTVs, validation_Torso = self.read_imape_path(self.validation_image_path)

        test_CTs, test_GTVs, test_Torso = self.read_imape_path(self.test_image_path)

        [depth, width, height] = self.return_depth_width_height(train_CTs)
        return train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
               test_CTs, test_GTVs, test_Torso, depth, width, height

    def return_depth_width_height(self, CTs):
        CT_image = sitk.ReadImage(''.join(CTs[int(0)]))
        CT_image = sitk.GetArrayFromImage(CT_image)
        return CT_image.shape[0], CT_image.shape[1], CT_image.shape[2]

    def return_normal_const(self, CTs):
        min_normal = 1E+10
        max_normal = -min_normal

        for i in range(len(CTs)):
            CT_image = sitk.ReadImage(''.join(CTs[int(i)]))
            CT_image = sitk.GetArrayFromImage(CT_image)
            max_tmp = np.max(CT_image)
            if max_tmp > max_normal:
                max_normal = max_tmp
            min_tmp = np.min(CT_image)
            if min_tmp < min_normal:
                min_normal = min_tmp
        return min_normal, max_normal

    # =================================================================
    def image_padding(self, img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img

    # =================================================================

    def read_image_seg_volume(self, CTs, GTVs, Torso, img_index, ct_cube_size, gtv_cube_size):

        CT_image1 = sitk.ReadImage(''.join(CTs[int(img_index)]))
        voxel_size = CT_image1.GetSpacing()
        origin = CT_image1.GetOrigin()
        direction = CT_image1.GetDirection()

        #
        CT_image = (CT_image1)  # /CT_image.mean()

        GTV_image = sitk.ReadImage(''.join(GTVs[int(img_index)]))
        #

        Torso_image = sitk.ReadImage(''.join(Torso[int(img_index)]))
        #

        padd_zero = 87 * 2 + 2
        crop = sitk.CropImageFilter()
        crop.SetLowerBoundaryCropSize([int(padd_zero / 2) + 1, int(padd_zero / 2) + 1,
                                       int(padd_zero / 2) + 1])
        crop.SetUpperBoundaryCropSize([int(padd_zero / 2), int(padd_zero / 2), int(padd_zero / 2)])
        CT_image = crop.Execute(CT_image)
        GTV_image = crop.Execute(GTV_image)
        Torso_image = crop.Execute(Torso_image)

        # padd:
        gtv = sitk.GetArrayFromImage(GTV_image)
        one = np.where(gtv)
        c_x = int((np.min(one[0]) + np.max(one[0])) / 2)
        c_y = int((np.min(one[1]) + np.max(one[1])) / 2)
        c_z = int((np.min(one[2]) + np.max(one[2])) / 2)

        if c_x - int(ct_cube_size / 2) < 0:
            xp1 = np.abs(c_x - int(ct_cube_size / 2))
        else:
            xp1 = int(ct_cube_size) - (c_x - int(2 * c_x / ct_cube_size) * int(ct_cube_size / 2))
        if c_x + int(ct_cube_size / 2) > np.shape(gtv)[0]:
            xp2 = int(ct_cube_size / 2) - (np.abs(np.shape(gtv)[0] - c_x))
        else:
            xp2 = int(ct_cube_size) - \
                  ((np.shape(gtv)[0] - c_x) -
                   int(2 * (np.shape(gtv)[0] - c_x) / ct_cube_size) * int(ct_cube_size / 2))

        if c_y - int(ct_cube_size / 2) < 0:
            yp1 = np.abs(c_y - int(ct_cube_size / 2))
        else:
            yp1 = ct_cube_size - (
            (c_y - int(ct_cube_size / 2)) - (int((c_y - int(ct_cube_size / 2)) / int(ct_cube_size)) * ct_cube_size))
        if c_y + int(ct_cube_size / 2) > np.shape(gtv)[1]:
            yp2 = np.abs((np.shape(gtv)[1] - c_y) - int(ct_cube_size / 2))
        else:
            yp2 = ct_cube_size - ((np.shape(gtv)[1] - (c_y + int(ct_cube_size / 2))) -
                                  np.floor(
                                      (np.shape(gtv)[1] - (c_y + int(ct_cube_size / 2))) / ct_cube_size) * ct_cube_size)

        if c_z - int(ct_cube_size / 2) < 0:
            zp1 = np.abs(c_z - int(ct_cube_size / 2))
        else:
            zp1 = ct_cube_size - (
            (c_z - int(ct_cube_size / 2)) - (int((c_z - int(ct_cube_size / 2)) / int(ct_cube_size)) * ct_cube_size))
        if c_z + int(ct_cube_size / 2) > np.shape(gtv)[2]:
            zp2 = np.abs((np.shape(gtv)[2] - c_z) - int(ct_cube_size / 2))
        else:
            zp2 = ct_cube_size - ((np.shape(gtv)[2] - (c_z + int(ct_cube_size / 2))) -
                                  np.floor(
                                      (np.shape(gtv)[2] - (c_z + int(ct_cube_size / 2))) / ct_cube_size) * ct_cube_size)



        CT_image = self.image_padding(img=CT_image,
                                      padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                      padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                      constant=-1024)
        GTV_image = self.image_padding(img=GTV_image,
                                       padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                       padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                       constant=0)
        Torso_image = self.image_padding(img=Torso_image,
                                         padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                         padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                         constant=0)

        # ----------------------------------------
        ct = sitk.GetArrayFromImage(CT_image)
        gtv1 = sitk.GetArrayFromImage(GTV_image)
        # Torso_image = sitk.GetArrayFromImage(Torso_image)
        c = 0
        gap = ct_cube_size - gtv_cube_size

        for _z in (
                range(int(ct_cube_size / 2) + 1, ct.shape[0] - int(ct_cube_size / 2) + 7,
                      int(ct_cube_size) - int(gap) + 1)):
            for _x in (range(int(ct_cube_size / 2) + 1, ct.shape[1] - int(ct_cube_size / 2) + 7,
                             int(ct_cube_size) - int(gap) + 1)):
                for _y in (range(int(ct_cube_size / 2) + 1, ct.shape[2] - int(ct_cube_size / 2) + 7,
                                 int(ct_cube_size) - int(gap) + 1)):
                    gtv = gtv1[_z - int(gtv_cube_size / 2) - 1:_z + int(gtv_cube_size / 2),
                          _x - int(gtv_cube_size / 2) - 1:_x + int(gtv_cube_size / 2),
                          _y - int(gtv_cube_size / 2) - 1:_y + int(gtv_cube_size / 2)]
                    if len(np.where(gtv[:, :, :] != 0)[0]):
                        print(len(np.where(gtv[:, :, :] != 0)[0]))
                        c = c + 1
        if c != 1:
            print('hhhhhhhhhhhhhhhhhhhhhhh')
        # ----------------------------------------

        CT_image = sitk.GetArrayFromImage(CT_image)
        GTV_image = sitk.GetArrayFromImage(GTV_image)
        Torso_image = sitk.GetArrayFromImage(Torso_image)

        return CT_image, GTV_image, Torso_image, GTV_image.shape[0], voxel_size, origin, direction

    # =================================================================
    def read_image(self, CT_image, GTV_image, img_height, img_padded_size, seg_size, depth):
        img = CT_image[depth, 0:img_height - 1, 0:img_height - 1]
        img1 = np.zeros((1, img_padded_size, img_padded_size))
        fill_val = img[0][0]
        img1[0][:][:] = np.lib.pad(img, (
            int((img_padded_size - img_height) / 2 + 1), int((img_padded_size - img_height) / 2 + 1)),
                                   "constant", constant_values=(fill_val, fill_val))
        img = img1[..., np.newaxis]
        seg1 = (GTV_image[depth, int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2),
                int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2)])
        seg = np.eye(2)[seg1]
        seg = seg[np.newaxis]
        return img, seg

    # =================================================================
    def check(self, GTVs, width_patch, height_patch, depth_patch):
        no_of_images = len(GTVs)
        for ii in range(no_of_images):
            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            if (max(depth_patch[ii]) > len(GTV_image)):
                print('error')

                # =================================================================


        # =================================================================

    def shuffle_lists(self, rand_width1, rand_height1, rand_depth1):
        index_shuf = list(range(len(rand_width1)))
        shuffle(index_shuf)
        rand_width11 = np.hstack([rand_width1[sn]]
                                 for sn in index_shuf)
        rand_depth11 = np.hstack([rand_depth1[sn]]
                                 for sn in index_shuf)
        rand_height11 = np.hstack([rand_height1[sn]]
                                  for sn in index_shuf)
        return rand_width11, rand_height11, rand_depth11



    # =================================================================

    def read_all_validation_batches(self, CTs, GTVs, total_sample_no, GTV_patchs_size, patch_window, img_width,
                                    img_height, epoch, img_padded_size, seg_size, whole_image=0):
        self.seed += 1
        np.random.seed(self.seed)

        if whole_image:
            # img_padded_size = 519
            # seg_size = 505

            ii = np.random.randint(0, len(CTs), size=1)
            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))

            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = (CT_image)  # /CT_image.mean()

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            GTV_max = GTV_image.max()

            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])



            rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=1)

            img = CT_image[rand_depth, 0:img_height - 1, 0:img_height - 1]

            img1 = np.zeros((1, img_padded_size, img_padded_size))
            fill_val = img[0][0][0]
            img1[0][:][:] = np.lib.pad(img[0], (
                int((img_padded_size - img_height) / 2 + 1), int((img_padded_size - img_height) / 2 + 1)),
                                       "constant", constant_values=(fill_val, fill_val))
            img = img1[..., np.newaxis]

            seg1 = (
            GTV_image[rand_depth, int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2),
            int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2)])
            seg = np.eye(2)[seg1]
            return img, seg


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
                CT_image = (CT_image)  # /CT_image.mean()
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
                                         ]).astype(int)
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

        return CT_image_patchs1, GTV_patchs1

    # =================================================================


    def read_data_all_train_batches(self, CTs, GTVs, total_sample_no, GTV_patchs_size, patch_window, img_width,
                                    img_height, epoch):
        self.seed+=1
        np.random.seed(self.seed)
        print("Reading %d training batches... " % (total_sample_no))
        len_CT = len(CTs)  # patients number
        sample_no = int(total_sample_no / len_CT)  # no of samples that must be selected from each patient
        if sample_no * len_CT < total_sample_no:  # if division causes to reduce total samples
            # remain=total_sample_no-sample_no*len_CT
            sample_no = sample_no + 2

        CT_image_patchs = []
        GTV_patchs = []
        for ii in range(len_CT):  # select samples from each patient:

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = (CT_image)  # /CT_image.mean()
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
                # print("c
                # ounter: %d" %(counter))
                dpth = np.random.randint(tumor_begin, tumor_end + 1, size=1)  # select one slice
                ones = np.where(
                    GTV_image[dpth, 0:img_width, 0:img_height] != 0)  # GTV indices of slice which belong to tumor
                if len(ones[0]):  # if not empty
                    tmp = int((sample_no * .5) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]), size=tmp)  # number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]), size=1)  # number of samples from each slice

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
                                          int(rand_height1[sn]) - int(patch_window / 2) - 1: int(rand_height1[sn]) +
                                                                                             int(patch_window / 2)])[
                                             ..., np.newaxis]
                                         for sn in range(len(rand_height1))])
            # print('2')

            GTV_patchs1 = np.stack([(GTV_image[
                                     int(rand_depth1[sn]),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(int)
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

                # print("length: %d" %(len(CT_image_patchs)))

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

        return GTV_patchs1, CT_image_patchs1