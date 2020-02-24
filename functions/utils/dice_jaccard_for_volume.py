"""
_dice.py : Dice coefficient for comparing set similarity.
"""
import SimpleITK as sitk
from scipy.spatial.distance import jaccard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from read_data import _read_data

eps=.00001
def dice(im1, im2):

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum()+eps)
def jaccard(im1,im2):


    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Jaccard coefficient
    intersection = np.logical_and(im1, im2)

    return  intersection.sum() / (im1.sum() + im2.sum()-intersection.sum()+eps)

def dicejacc():
    _rd = _read_data()
    test_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/prostate_test/'
    result_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_14/densenet_unet_output_volume_88888_0.5_2/'
    test_CTs, test_GTVs = _rd.read_imape_path(test_path)


    xtickNames = []
    dice_boxplot=[]
    jacc_boxplot=[]
    for i in range(len(test_GTVs)):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        d = []
        j = []
        [CT_image, GTV_image, volume_depth, voxel_size, origin, direction] = _rd.read_image_seg_volume(test_CTs,
                                                                                                       test_GTVs,
                                                                                                       i)
        ss = str(test_CTs[i]).split("/")
        name = ss[8] + '_' + ss[9]
        xtickNames.append(name)
        res = sitk.ReadImage(result_path + name + '.mha')
        res = sitk.GetArrayFromImage(res)
        for jj in range(GTV_image.shape[0]):
            im1 = np.asarray(res[jj][:][:]).astype(np.bool)
            im2 = np.asarray(GTV_image[jj][:][:]).astype(np.bool)
            im2 = im2[0:511,0:511]
            im2 = im2[int(511/2)-int(im1.shape[0]/2)-1:int(511/2)+int(im1.shape[0]/2),
                      int(511 / 2) - int(im1.shape[1]/2) - 1:int(511 / 2) + int(im1.shape[1]/2)]
            dice(im1, im2)
            d.append(dice(im1, im2))
            j.append(jaccard(im1, im2))

        index = np.arange(len(d))
        bar_width = 0.35
        opacity = 1
        rects1 =ax1.bar(index, d, bar_width,
                            alpha=1,
                            color='b',
                            label='Dice')#'bo--',range(2), j, 'rs--')
        rects2 =ax2.bar(index+bar_width, j,  bar_width,
                            alpha=1,
                            color='r',
                            label='Jaccard')#,range(2), j, 'rs--')
        # first_legend = plt.legend(handles=[line1], loc=1)
        # ax = plt.gca().add_artist(first_legend)
        ax1.set_xlabel('Slices')
        ax2.set_xlabel('Slices')
        ax1.set_ylabel('Accuracy')
        ax2.set_ylabel('Accuracy')
        non_zero_j =  [x for x in j if x != 0]
        non_zero_d =  [x for x in d if x != 0]
        if len(non_zero_j)!=0:
            title2=name+': '+'jaccard: (%.2f,%.2f)' %\
                        (min(non_zero_j), max(non_zero_j))
            title1 = name + ': ' + 'dice: (%.2f,%.2f)' % \
                                   ( min(non_zero_d), max(non_zero_d))
        else:
            title2 = name + ': ' + 'jaccard: (%.2f,%.2f)' % \
                                  (min(j), max(j))
            title1 = name + ': ' + 'dice: (%.2f,%.2f)' % \
                                  ( min(d), max(d))
        fig1.suptitle(title1)
        ax1.legend( loc=4)
        plt.ylim(0, 1.1)
        fig1.savefig(result_path + '/dice_'+name+'.png')

        fig2.suptitle(title2)
        ax2.legend(loc=4)
        plt.ylim(0, 1.0)
        fig2.savefig(result_path + '/jaccard_'+name+'.png')


        dice_boxplot.append((non_zero_d))
        jacc_boxplot.append((non_zero_j))

        fig1.clear()
        fig2.clear()
    fig, ax = plt.subplots()
    plt.boxplot(dice_boxplot, 0, '')
    plt.xticks(list(range(1, len(dice_boxplot) + 1)), xtickNames, rotation='vertical')
    plt.ylabel('Dice')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=.4)
    fig.savefig(result_path + '/dice_boxplot.png')

    fig, ax = plt.subplots()
    plt.boxplot(jacc_boxplot, 0, '')
    plt.xticks(list(range(1, len(jacc_boxplot) + 1)), xtickNames, rotation='vertical')
    plt.ylabel('Jaccard')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=.4)
    fig.savefig(result_path + '/jacc_boxplot.png')



# plt.plot(i, d, 'r--', i, j, 'bs')
# plt.show()

dicejacc()


