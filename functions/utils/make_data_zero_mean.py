#this is ok

from os import listdir

# import math as math
import numpy as np
from os.path import isfile, join


class perprocess:
    def __init__(self):
        print("ínit")
        self.resampled_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/Esophagus_raw/'
        path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/'

        self.prostate_ext_gt = 'Contours_Cleaned/'
        self.startwith_4DCT = '4DCT_'
        self.startwith_GTV = 'GTV_'
        self.prostate_ext_gt = 'Contours_Cleaned/'
        self.prostate_ext_img = 'Images/'

        self.resample_tag = ''
        self.img_name = 'CT' + self.resample_tag + '.mha'
        self.label_name1 = 'GTV_CT' + self.resample_tag + '.mha'
        self.label_name2 = 'GTV_prim' + self.resample_tag + '.mha'
        self.torso_tag = 'CT_Torso' + self.resample_tag + '.mha'
        self.startwith_4DCT = '4DCT_'
        self.startwith_GTV = 'GTV_'
    def read_data_path(self):
        data_dir = [join(self.resampled_path, f) for f in listdir(self.resampled_path) if ~isfile(join(self.resampled_path, f))]
        data_dir=np.sort(data_dir)

        train_CTs,train_GTVs,train_Torso=self.read_train_data(np.hstack((data_dir[0:15],data_dir[21:36])))
        validation_CTs,validation_GTVs,validation_Torso=self.read_train_data(np.hstack((data_dir[18:21],data_dir[36:39])))
        test_CTs,test_GTVs,test_Torso=self.read_train_data(np.hstack((data_dir[15:18],data_dir[39:50])))
        #already read all the path, now select train, validation and test!

        return train_CTs,train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
               test_CTs, test_GTVs, test_Torso

        # ========================
    def read_train_data(self, data_dir):
        image_path = self.resampled_path
        CTs = []
        GTVs = []
        Torsos = []
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
                if (len(GTV_path) == 0):
                    GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                f.endswith(self.label_name2)]

                CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                             f.startswith(self.startwith_4DCT) & f.endswith('%' + self.resample_tag + '.mha')]
                CT_path = CT_path + CT4D_path  # here sahar
                for i in range(len(CT4D_path)):
                    percent = CT4D_path[i].split('/')[-1].split('.')[0].split('_')[1]
                    name_gtv4d = 'GTV_4DCT_' + percent + self.resample_tag + '.mha'
                    GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                    Torso_gtv4d = self.startwith_4DCT + percent + '_Torso' + self.resample_tag + '.mha'
                    Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))

                CTs += (CT_path)
                GTVs += (GTV_path)
                Torsos += (Torso_path)
        return CTs, GTVs, Torsos

if __name__ == "__main__":
    prepro=perprocess()
    # densnet_unet_config = [3, 4, 4, 4, 3]
    compression_coefficient =.5
    growth_rate = 4
    # pad_size = 14
    # ext = ''.join(map(str, densnet_unet_config))  # +'_'+str(compression_coefficient)+'_'+str(growth_rate)
    data=2
    densnet_unet_config=[3,4,4,4,3]

    train_tag='train/'
    validation_tag='validation/'
    test_tag='Esophagus_raw/'


    img_name = 'CT_padded.mha'
    label_name = 'GTV_prim_padded.mha'
    torso_tag = 'CTpadded.mha'



    # test_CTs, test_GTVs ,test_Torsos= _rd.read_imape_path(test_path)
    train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
    test_CTs, test_GTVs, test_Torso = prepro.read_data_path()

    test_CTs=np.sort(test_CTs)
    test_GTVs=np.sort(test_GTVs)
    test_Torso=np.sort(test_Torso)
