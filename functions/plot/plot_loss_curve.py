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

    log_name_train_type = {'Training': 'train12521_1_4_all_01/',
                                   'Validation': 'validation12521_1_4_all_01/'}


    train_mode_list = ['Training','Validation']#,
    tag_list = ['gradients/avegare_perfusion', 'gradients/perfusion',
                 'gradients/avegare_angiography','gradients/angiography',
                'Loss/ave_loss', 'Loss/ave_loss_angio', 'Loss/ave_loss_perf']
    g_ave_perf_tr = []
    g_perf_tr = []
    g_ave_angio_tr = []
    g_angio_tr = []
    loss_tr = []
    loss_a_ave_tr = []
    loss_p_ave_tr = []

    loss_vl = []
    loss_a_ave_vl = []
    loss_p_ave_vl = []
    for train_mode in train_mode_list:
        # log_folder='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/debug_Log/synth-7-shark/'
        log_folder='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2018-08-15/cross_validation/13331_0.75_4-cross-NoRand-tumor25-004/'
        log_test_folder = log_folder + log_name_train_type[train_mode]

        loss_dict = dict()
        file_list = [f for f in os.listdir(log_test_folder) if os.path.isfile(os.path.join(log_test_folder, f))]

        for file in file_list:


            for e in tf.train.summary_iterator(log_test_folder + file):
                broken_point= 30050 #27725 for shark8, 30050 for shark7
                # if e.step==broken_point:
                #     break
                for v in e.summary.value:
                    # print(v.tag)
                    if train_mode=='Training':
                        if v.tag == tag_list[0]:
                            g_ave_perf_tr.append(v.simple_value/25.)
                        elif v.tag == tag_list[1]:
                            g_perf_tr.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            g_ave_angio_tr.append(v.simple_value/25.)
                        elif v.tag == tag_list[3]:
                            g_angio_tr.append(v.simple_value)
                        elif v.tag == tag_list[4]:
                            loss_tr.append(v.simple_value)
                        elif v.tag == tag_list[5]:
                            loss_a_ave_tr.append(v.simple_value)
                        elif v.tag == tag_list[6]:
                            loss_p_ave_tr.append(v.simple_value)
                    else:
                        if v.tag == tag_list[4]:
                            loss_vl.append(v.simple_value)
                        elif v.tag == tag_list[5]:
                            loss_a_ave_vl.append(v.simple_value)
                        elif v.tag == tag_list[6]:
                            loss_p_ave_vl.append(v.simple_value)

    rng=list(range(0, 25 * len(g_ave_perf_tr), 25))
    rng_vl=list(range(0, 125 * len(loss_vl), 125))
    # 8cffdb,137e6d
    #==================================


    fig = plt.figure()

    gs = GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    plt.plot(rng, smooth_curve(g_ave_perf_tr, 1), c='#fed0fc', alpha=.6)
    plt.plot(rng, smooth_curve(g_ave_perf_tr, 20), c='#c875c4', label='Perfusion')
    plt.ylabel('Gradient magnitude of loss')
    ax1.legend()
    # ax2= ax1.twinx()
    ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
    plt.plot(rng, smooth_curve(g_ave_angio_tr, 1), c='#c1c6fc', alpha=.6)
    plt.plot(rng, smooth_curve(g_ave_angio_tr, 20), c='#4b57db', label='Angiography')
    plt.ylabel('Gradient magnitude of loss')
    ax2.legend()
    plt.yscale('symlog', linthreshy=0.0001)

    ax3 = fig.add_subplot(gs[1, 0])
    plt.plot(rng, loss_tr, c='#ff000d', label='Training')
    plt.plot(rng_vl, loss_vl, linestyle='--', c='#916e99', label='Validation')

    plt.ylabel('Perfusion Huber loss')
    ax3.legend()
    ax4 = fig.add_subplot(gs[1, 1])
    # ax4 = ax3.twinx()
    plt.plot(rng, loss_a_ave_tr, c='#fd4659', label='Training')
    plt.plot(rng_vl, loss_a_ave_vl, linestyle='--', c='#2a7e19', label='Validation')

    plt.ylabel('Angiography Huber loss')
    plt.yscale('symlog', linthreshy=0.0001)
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, :])  # Second row, span all columns
    plt.plot(rng, loss_p_ave_tr, c='#0e87cc', label='Training')
    plt.plot(rng_vl, loss_p_ave_vl, linestyle='--', c='#cb0015', label='Validation')
    plt.ylabel('Total Huber loss')
    plt.xlabel('Iterations')
    plt.subplots_adjust(hspace=.2)
    ax5.legend()

    plt.show()


    # #================================
    # fig = plt.figure()
    # ax = plt.subplot(211)
    # plt.plot(rng, smooth_curve(g_ave_perf_tr, 1), c='#fed0fc', alpha=.6)
    # plt.plot(rng, smooth_curve(g_ave_perf_tr, 20), c='#c875c4',label='Perfusion' )
    # plt.title('Gradient loss')
    # ax.legend()
    #
    # # ==================================
    # # plt.figure()
    # ax = plt.subplot(212)
    # plt.plot(rng, smooth_curve(g_ave_angio_tr, 1), c='#c1c6fc', alpha=.6)
    # plt.plot(rng, smooth_curve( g_ave_angio_tr, 20), c='#4b57db',label='Angiography')
    # # plt.title('Gradient loss')
    # ax.legend()
    # # ==================================
    # plt.figure()
    # ax = plt.subplot(311)
    # plt.plot(rng,  loss_tr, c='#ff000d',label='Train')
    # plt.plot(rng_vl, loss_vl,linestyle='--', c='#916e99',label='Validation' )
    # plt.title('Huber loss function')
    # ax.legend()
    # # ==================================
    # # plt.figure()
    # ax = plt.subplot(312)
    # plt.plot(rng, loss_a_ave_tr, c='#fd4659', label='Train')
    # plt.plot(rng_vl, loss_a_ave_vl, linestyle='--', c='#004577', label='Validation')
    # plt.title('Angiography')
    # ax.legend()
    # # ==================================
    # # plt.figure()
    # ax = plt.subplot(313)
    # plt.plot(rng, loss_p_ave_tr, c='#0e87cc', label='Train')
    # plt.plot(rng_vl, loss_p_ave_vl, linestyle='--', c='#cb0015', label='Validation')
    # plt.title('Perfusion')
    # ax.legend()
    # # ==================================
    # plt.show()
    #
    # print('file')

