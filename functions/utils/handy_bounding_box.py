import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from read_data import _read_data
import SimpleITK as sitk
import numpy as np
import pandas as pd


class handy_bounding_box:
    def __init__(self):

        self.train_tag='train/'
        self.validation_tag='validation/'
        self.test_tag='test/'
        self.img_name='CT_padded.mha'
        self.label_name='GTV_CT_padded.mha'
        self.torso_tag='CT_padded_Torso.mha'
        self.data=2
        self.gtv_min_z = []
        self.gtv_max_z = []
        self.gtv_min_x = []
        self.gtv_max_x = []
        self.gtv_min_y = []
        self.gtv_max_y = []
        self.mask_min_z = []
        self.mask_max_z = []
        self.mask_min_x = []
        self.mask_max_x = []
        self.mask_min_y = []
        self.mask_max_y = []
        self._rd = _read_data(data=self.data, train_tag=self.train_tag, validation_tag=self.validation_tag,
                     test_tag=self.test_tag,
                     img_name=self.img_name, label_name=self.label_name, torso_tag=self.torso_tag)
    def compute_margins(self):
        '''read path of the images for train, test, and validation'''
        train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
        test_CTs, test_GTVs, test_Torso = self._rd.read_data_path()

        train_CTs=train_CTs+validation_CTs
        train_CTs=train_CTs+test_CTs

        train_GTVs=train_GTVs+validation_GTVs
        train_GTVs=train_GTVs+test_GTVs

        train_Torso=train_Torso+validation_Torso
        train_Torso=train_Torso+test_Torso


        for i in range(len(train_CTs)):
            gtv=sitk.GetArrayFromImage(sitk.ReadImage(''.join(train_GTVs[i])))
            mask=sitk.GetArrayFromImage(sitk.ReadImage(''.join(train_Torso[i])))
            # if i==175:
            #     print('s')
            self.gtv_min_z.append(np.min(np.where(gtv)[0]))
            self.gtv_max_z.append(np.max(np.where(gtv)[0]))

            self.gtv_min_x .append(np.min(np.where(gtv)[1]))
            self.gtv_max_x .append( np.max(np.where(gtv)[1]))

            self.gtv_min_y.append( np.min(np.where(gtv)[2]))
            self.gtv_max_y.append( np.max(np.where(gtv)[2]))

            self.mask_min_z.append( np.min(np.where(mask)[0]))
            self.mask_max_z.append( np.max(np.where(mask)[0]))

            self.mask_min_x.append( np.min(np.where(mask)[1]))
            self.mask_max_x.append( np.max(np.where(mask)[1]))

            self.mask_min_y.append( np.min(np.where(mask)[2]))
            self.mask_max_y.append( np.max(np.where(mask)[2]))

            print(i)




    def write_xls(self,path):
        import xlwt
        book_ro = xlwt.Workbook()
        ws = book_ro.add_sheet('sheet' + str(1))
        ws.write(0, 0, 'gtv_min_z')
        ws.write(0, 1, 'gtv_max_z')

        ws.write(0, 2, 'gtv_min_x')
        ws.write(0, 3, 'gtv_max_x')

        ws.write(0, 4, 'gtv_min_y')
        ws.write(0, 5, 'gtv_max_y')

        ws.write(0, 6, 'mask_min_z')
        ws.write(0, 7, 'mask_max_z')

        ws.write(0, 8, 'mask_min_x')
        ws.write(0, 9, 'mask_max_x')

        ws.write(0, 10, 'mask_min_y')
        ws.write(0, 11, 'mask_max_y')
        for i in range(len(self.gtv_min_z)):
            ws.write(i + 1, 0, '{0:.8f}'.format(self.gtv_min_z[i]))
            ws.write(i + 1, 1, '{0:.8f}'.format(self.gtv_max_z[i]))

            ws.write(i + 1, 2, '{0:.8f}'.format(self.gtv_min_x[i]))
            ws.write(i + 1, 3, '{0:.8f}'.format(self.gtv_max_x[i]))

            ws.write(i + 1, 4, '{0:.8f}'.format(self.gtv_min_y[i]))
            ws.write(i + 1, 5, '{0:.8f}'.format(self.gtv_max_y[i]))

            ws.write(i + 1, 6, '{0:.8f}'.format(self.mask_min_z[i]))
            ws.write(i + 1, 7, '{0:.8f}'.format(self.mask_max_z[i]))

            ws.write(i + 1, 8, '{0:.8f}'.format(self.mask_min_x[i]))
            ws.write(i + 1, 9, '{0:.8f}'.format(self.mask_max_x[i]))

            ws.write(i + 1, 10, '{0:.8f}'.format(self.mask_min_y[i]))
            ws.write(i + 1, 11, '{0:.8f}'.format(self.mask_max_y[i]))
        book_ro.save(path+'.xls')

    def read_xls(self,path):
        df = pd.read_excel(path+'.xls', sheetname='sheet1')
        self.gtv_min_z = df['gtv_min_z']
        self.gtv_max_z = df['gtv_max_z']
        self.gtv_min_x = df['gtv_min_x']
        self.gtv_max_x = df['gtv_max_x']
        self.gtv_min_y = df['gtv_min_y']
        self.gtv_max_y = df['gtv_max_y']
        self.mask_min_z = df['mask_min_z']
        self.mask_max_z = df['mask_max_z']
        self.mask_min_x = df['mask_min_x']
        self.mask_max_x = df['mask_max_x']
        self.mask_min_y = df['mask_min_y']
        self.mask_max_y = df['mask_max_y']
    def show_diag(self,gtv_min,gtv_max,mask_min,mask_max):
        fig9 = plt.figure()
        ax9 = fig9.add_subplot(111, aspect='equal')

        for i in range(len(y1)):
            for p in ([
                patches.Rectangle(
                    (y1[i], gtv_min[i]), 5, (gtv_max[i] - gtv_min[i]), fill=False,
                    linestyle='solid', color='green'
                ),
                patches.Rectangle(
                    (y1[i] + .5, mask_min[i]), 4, (mask_max[i] - mask_min[i]), fill=False,
                    linestyle='dashed', color='red'
                ),

            ]):
                ax9.add_patch(p)
        # plt.xticks([r + 10 for r in range(1, 850, 10)], list(img_nm), rotation=90, fontsize=10)
        plt.ylim(-5, 500)
        plt.xlim(-5, 6000)
        plt.subplots_adjust(bottom=0.35)

        red_patch = mpatches.Patch(color='red', label='DenseUnet result')
        green_patch = mpatches.Patch(color='green', label='GTV')

        # plt.legend([red_patch, (red_patch, green_patch)], ["DenseUnet result", "GTV"])

if __name__ == '__main__':
    path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log-28-1-2018/19-3-2018-more-analysis/boundbox'
    hbb=handy_bounding_box()
    # hbb.compute_margins()
    # hbb.write_xls(path)
    hbb.read_xls(path)
    y1 = list(range(10, 5540, 10))
    # y1 = list(range(10, 50, 10))

    # fig9 = plt.figure()
    # ax9 = fig9.add_subplot(111, aspect='equal')
    # y1 = [1, 2, 3]
    # end_gtv = [10, 15, 16]
    # start_gtv = [4, 6, 9]
    #
    # end_result = [18, 20, 23]
    # start_result = [10, 15, 13]

    # for i in range(len(y1)):
    #     for p in ([
    #         patches.Rectangle(
    #             (y1[i], hbb.gtv_min_z[i]), 5, (hbb.gtv_max_z[i] - hbb.gtv_min_z[i]), fill=False,
    #             linestyle='solid', color='green'
    #         ),
    #         patches.Rectangle(
    #             (y1[i] + .5, hbb.mask_min_z[i]), 4, (hbb.mask_max_z[i] - hbb.mask_min_z[i]), fill=False,
    #             linestyle='dashed', color='red'
    #         ),
    #
    #     ]):
    #         ax9.add_patch(p)
    # # plt.xticks([r + 10 for r in range(1, 850, 10)], list(img_nm), rotation=90, fontsize=10)
    # plt.ylim(-5, 500)
    # plt.xlim(-5, 5540)
    # plt.subplots_adjust(bottom=0.35)
    hbb.show_diag(hbb.gtv_min_z,hbb.gtv_max_z,hbb.mask_min_z,hbb.mask_max_z)
    print('z_min:%d, z_max:%d'%(np.min(hbb.gtv_min_z),np.max(hbb.gtv_max_z)))

    hbb.show_diag(hbb.gtv_min_x, hbb.gtv_max_x, hbb.mask_min_x, hbb.mask_max_x)
    print('x_min:%d, x_max:%d' % (np.min(hbb.gtv_min_x), np.max(hbb.gtv_max_x)))

    hbb.show_diag(hbb.gtv_min_y, hbb.gtv_max_y, hbb.mask_min_y, hbb.mask_max_y)
    print('y_min:%d, y_max:%d' % (np.min(hbb.gtv_min_y), np.max(hbb.gtv_max_y)))
    print('f')



