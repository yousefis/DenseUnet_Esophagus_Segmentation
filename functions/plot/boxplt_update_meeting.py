import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def hgrid(axes,top,bottom,major_gap,minor_gap,axis):
    axes.set_ylim(bottom, top)
    major_ticks = np.arange(bottom, top, major_gap)
    minor_ticks = np.arange(bottom, top, minor_gap)
    axes.set_yticks(major_ticks)
    axes.set_yticks(minor_ticks, minor=True)
    # And a corresponding grid
    # axes[0].grid(which='both')
    # Or if you want different settings for the grids:
    axes.grid(which='minor', alpha=0.05,axis=axis)
    axes.grid(which='major', alpha=0.5,axis=axis)


def fill_pb_color(bplots,colors,axes):
    for bplot in bplots:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    axes.legend([bplot2["boxes"][0],# bplot2["boxes"][1],
                 #bplot2["boxes"][2], bplot2["boxes"][3]
    ],
                   cnn_tags, loc='right', bbox_to_anchor=(1.2, -0.09),
                   fancybox = True, shadow = True, ncol = 5)

def plot_bp(axes,title,data):
    bplot = axes.boxplot(data,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             # labels=cnn_tags,
                             showmeans=True,
                             )  # will be used to label x-ticks
    axes.set_title(title)
    return bplot


def cumulative_dice(axes,cnn_tags,data,title):
    vec = np.zeros([len(data),11])
    # j=0
    for i in reversed(range(0, 11, 1)):
        for dc in range(len(data)):
            vec[ dc,i] = sum(data[dc] < (i / 10))
        # j=j+1
    y_labels = [ '0.0$\leq$', '0.1$\leq$','0.2$\leq$','0.3$\leq$','0.4$\leq$', '0.5$\leq$',  '0.6$\leq$',
                 '0.7$\leq$','0.8$\leq$','0.9$\leq$', '1.0$\leq$'
                ]
    df = pd.DataFrame({cnn_tags[0]: vec[0],
                      # cnn_tags[1]: vec[1],
                       #cnn_tags[2]: vec[2],
                       #cnn_tags[3]: vec[3],
                       }, index=y_labels)
    # df.plot(ax=axes)
    # axes = df.plot.barh(color=colors)
    bplot=df.plot(kind='bar', legend=False, ax=axes,color=colors,title=title,rot=-300)
    return bplot

def read_xls(xls_path,results,parent_pth,fields):
    without_lc=[]
    with_lc=[]
    for xp in xls_path:
        p=parent_pth+xp+results+'all_dice2.xlsx'
        sheet_name = 'Sheet1'
        xl_file = pd.ExcelFile(p)
        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}
        without_lc.append(dfs['Sheet1'][fields[0]])
        with_lc.append(dfs['Sheet1'][fields[1]])
    return without_lc,with_lc
#this file shows the boxplots for the journal paper
if __name__=='__main__':
    parent_pth='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/'
    xls_path=[
        # '33533_0.75_4-train1-04252020_220/',  #dice+Nodistancemap
              # '33533_0.75_4-train1-07032020_170/', #dice+distancemap+attebtion channel
              # '33533_0.75_4-train1-07052020_000/',  #dice+attention channel no distancemap
              '33533_0.75_4-train1-07142020_020/',  #dice+attention spatial  no distancemap
              # '33533_0.75_4-train1-07102020_140/',  # dice+attention channel+spatial  no distancemap
              ]

    cnn_tags=[
              # 'DilatedDenseUnet',
              # 'DilatedDenseChannelAttentionUnet',
              'DilatedDenseSpatialAttentionUnet',
              # 'DilatedDenseChannelSpatialAttentionUnet'
        ]
    test_vali=0
    if test_vali==0:
        results='result/'
    else:
        results='result_vali/'
    #=============================================
    #read xls files and fill vectors
    dcs_without_lc,dcs_with_lc= read_xls(xls_path, results, parent_pth, fields= ['DCS-LC', 'DCS+LC'])
    #plot boxplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1= plot_bp(axes[0], 'Without largest component',dcs_without_lc)
    bplot2= plot_bp(axes[1], 'With largest component',dcs_with_lc)
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen','orchid']
    fill_pb_color((bplot1,bplot2), colors, axes[1])
    hgrid(axes[0],top=1,bottom=0,major_gap=.1,minor_gap=0.05,axis='y')
    hgrid(axes[1],top=1,bottom=0,major_gap=.1,minor_gap=0.05,axis='y')
    fig.suptitle('DSC', fontsize=16)
    # =============================================
    # read xls files and fill vectors
    msd_without_lc, msd_with_lc = read_xls(xls_path, results, parent_pth, fields=['msd', 'msd_lc'])
    # plot boxplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = plot_bp(axes[0], 'Without largest component', msd_without_lc)
    bplot2 = plot_bp(axes[1], 'With largest component', msd_with_lc)
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    fill_pb_color((bplot1, bplot2), colors, axes[1])
    hgrid(axes[0],top=100,bottom=0,major_gap=50,minor_gap=25,axis='y')
    hgrid(axes[1],top=100,bottom=0,major_gap=50,minor_gap=25,axis='y')
    fig.suptitle('Mean surface distance', fontsize=16)
    # =============================================
    # # read xls files and fill vectors
    hd_without_lc, hd_with_lc = read_xls(xls_path, results, parent_pth, fields=['95hd', '95hd_lc'])
    # plot boxplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = plot_bp(axes[0], 'Without largest component', hd_without_lc)
    bplot2 = plot_bp(axes[1], 'With largest component', hd_with_lc)
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    fill_pb_color((bplot1, bplot2), colors, axes[1])
    hgrid(axes[0], top=250, bottom=0, major_gap=50, minor_gap=25,axis='y')
    hgrid(axes[1], top=150, bottom=0, major_gap=50, minor_gap=25,axis='y')
    fig.suptitle('95% Hausdorff distance', fontsize=16)

    # =============================================
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    # axes[0].invert_xaxis()
    # axes[0].yaxis.tick_right()
    cumulative_dice( axes[0], cnn_tags, data=dcs_without_lc,title='Without largest component')
    cumulative_dice( axes[1], cnn_tags, data=dcs_with_lc,title='With largest component')
    bp1=hgrid(axes[0], top=300, bottom=0, major_gap=100, minor_gap=20,axis='y')
    bp2=hgrid(axes[1], top=300, bottom=0, major_gap=100, minor_gap=20,axis='y')
    # axes.legend([bp2["boxes"][0], bp2["boxes"][1], bp2["boxes"][2], bp2["boxes"][3]],
    #             cnn_tags, loc='right', bbox_to_anchor=(1.2, -0.09),
    #             fancybox=True, shadow=True, ncol=5)

    plt.show()
    print(2)
    # adding horizontal grid lines

