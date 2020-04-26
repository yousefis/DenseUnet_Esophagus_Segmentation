from matplotlib.gridspec import GridSpec
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from matplotlib.font_manager import FontProperties

def smooth_curve(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
if __name__=='__main__':
    fontP = FontProperties()
    fontP.set_size('small')

    log_name_train_type = {'Training': 'train/',
                                   'Validation': 'validation/'}
    train_mode_list = ['Training','Validation']#,
    tag_list = ['cost_1',
    'dice_tumor',
    'average_validation_accuracy',
    'average_validation_loss',
    'average_dsc_loss',
    'accuracy_1',
    ]
    cost_1_tr = []
    dice_tumor_tr = []
    average_validation_accuracy_tr = []
    average_validation_loss_tr = []
    average_dsc_loss_tr = []
    accuracy_1_tr = []

    cost_1_vl = []
    dice_tumor_vl = []
    average_validation_accuracy_vl = []
    average_validation_loss_vl = []
    average_dsc_loss_vl = []
    accuracy_1_vl = []
    for train_mode in train_mode_list:
        log_folder='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-04162020_120/'
        log_test_folder = log_folder + log_name_train_type[train_mode]

        loss_dict = dict()
        file_list = [f for f in os.listdir(log_test_folder) if os.path.isfile(os.path.join(log_test_folder, f))]

        for file in file_list:


            for e in tf.train.summary_iterator(log_test_folder + file):
                broken_point= 30000 #27725 for shark8, 30050 for shark7
                if e.step==broken_point:
                    break
                for v in e.summary.value:
                    # print(v.tag)
                    if train_mode=='Training':
                        if v.tag == tag_list[0]:
                            cost_1_tr.append(v.simple_value/25.)
                        elif v.tag == tag_list[1]:
                            dice_tumor_tr.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            average_validation_accuracy_tr.append(v.simple_value/25.)
                        elif v.tag == tag_list[3]:
                            average_validation_loss_tr.append(v.simple_value)
                        elif v.tag == tag_list[4]:
                            average_dsc_loss_tr.append(v.simple_value)
                        elif v.tag == tag_list[5]:
                            accuracy_1_tr.append(v.simple_value)
                    else:
                        if v.tag == tag_list[0]:
                            cost_1_vl.append(v.simple_value/25.)
                        elif v.tag == tag_list[1]:
                            dice_tumor_vl.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            average_validation_accuracy_vl.append(v.simple_value/25.)
                        elif v.tag == tag_list[3]:
                            average_validation_loss_vl.append(v.simple_value)
                        elif v.tag == tag_list[4]:
                            average_dsc_loss_vl.append(v.simple_value)
                        elif v.tag == tag_list[5]:
                            accuracy_1_vl.append(v.simple_value)

    # rng=list(range(0, 50 * len(cost_1_tr), 50))
    rng=list(range(141,  len(cost_1_tr)))
    # rng_vl=list(range(0, 75 * len(cost_1_vl), 75))
    rng_vl=list(range(1,  142*len(cost_1_vl),142))
    # 8cffdb,137e6d
    #==================================


    fig = plt.figure()

    gs = GridSpec(1, 1)

    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    plt.plot(rng, smooth_curve(average_validation_loss_tr[140:-1], 1), c='#ff000d', alpha=.6)
    plt.plot(rng, smooth_curve(average_validation_loss_tr[140:-1], 20), c='#ff0001', alpha=.6)
    plt.plot(rng_vl, smooth_curve(average_validation_loss_vl[1:-1], 1), c='#916e99', alpha=.6)
    # plt.plot(rng, smooth_curve(cost_1_tr, 20), c='#c875c4', label='Perfusion')
    ax1.legend()
    plt.ylabel('loss')
    #

    # ax3 = fig.add_subplot(gs[1, 0])
    # plt.plot(rng, average_validation_loss_tr, c='#ff000d', label='Training')
    # plt.plot(rng_vl, average_validation_loss_vl, linestyle='--', c='#916e99', label='Validation')
    # ax3.legend()
    plt.show()



