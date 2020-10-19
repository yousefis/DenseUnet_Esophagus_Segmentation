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

def plot_bp(axes,title,data):
    bplot = axes.boxplot(data,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             # labels=cnn_tags,
                             showmeans=True,
                             )  # will be used to label x-ticks
    axes.set_title(title, pad=20)
    return bplot


def cumulative_dice(axes,cnn_tags,data,title):
    vec = np.zeros([len(data),11])
    # j=0
    for i in reversed(range(0, 11, 1)):
        for dc in range(len(data)):
            vec[ dc,i] = sum(data[dc] < (i / 10))

    y_labels = [ '0.0$\leq$', '0.1$\leq$','0.2$\leq$','0.3$\leq$','0.4$\leq$', '0.5$\leq$',  '0.6$\leq$',
                 '0.7$\leq$','0.8$\leq$','0.9$\leq$', '1.0$\leq$'
                ]
    df = pd.DataFrame({cnn_tags[0]: vec[0]/171*100,
                       cnn_tags[1]: vec[1]/171*100,
                       cnn_tags[2]: vec[2]/171*100,
                       cnn_tags[3]: vec[3]/171*100,
                       cnn_tags[4]: vec[4]/171*100,
                       # cnn_tags[5]: vec[5],
                       # cnn_tags[6]: vec[6],
                       }, index=y_labels)
    # df.plot(ax=axes)
    # axes = df.plot.barh(color=colors)
    bplot=df.plot(kind='bar', legend=False, ax=axes,color=colors,rot=-350)
    bplot.set_title(label=title, pad=20, )
    return bplot

def read_xls(xls_path,results,parent_pth,fields):
    without_lc=[]
    with_lc=[]
    for xp in xls_path:
        p=parent_pth+xp+results+'all_dice2.xlsx'
        print(parent_pth+xp+results)
        sheet_name = 'Sheet1'
        xl_file = pd.ExcelFile(p)

        df = pd.read_excel(p)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        print(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum())

        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}
        without_lc.append(dfs[sheet_name][fields[0]])
        with_lc.append(dfs[sheet_name][fields[1]])
    return without_lc,with_lc
#this file shows the boxplots for the journal paper
if __name__=='__main__':
    parent_pth='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/'
    xls_path=[#'33533_0.75_4-train1-04172020_140/',#DUnet
        # '33533_0.75_4-train1-05082020_090/',  #dice  Nodistancemap

              # '33533_0.75_4-train1-07052020_000/',  #dice+attention channel no distancemap
              # '33533_0.75_4-train1-07142020_020/',  #dice+attention spatial  no distancemap
              # '33533_0.75_4-train1-07102020_140/',  # dice+attention channel+spatial  no distancemap
              # '33533_0.75_4-train1-08132020_120/',
              '33533_0.75_4-train1-08242020_1950240/' , # dice+attention spatial  no distancemap +channel skip att+bourndry loss
            '33533_0.75_4-train1-07032020_170/',  # dice+distancemap+attebtion channel
              '33533_0.75_4-train1-08132020_10590/', #spatial only focal dice
              '/33533_0.75_4-train1-08132020_120/', #spatial only focal dice+surface
              '33533_0.75_4-train1-08052020_140/',  # dice+attention spatial  no distancemap +bourndry loss
              ]

    cnn_tags = ['DSC','DSC + DM',
                'DSC + Focal',
                'DSC + Focal + BL',
                'DSC + BL',
                ]

    test_vali=1
    if test_vali==0:
        results='result/'
        top=300
        top1=200
    else:
        results='result_vali/'
        top = 200
        top1 = 30

    #=============================================

    #read xls files and fill vectors
    dcs_without_lc,dcs_with_lc= read_xls(xls_path, results, parent_pth, fields= ['DCS-LC', 'DCS+LC'])
    #plot boxplots

    col = 2
    axes = [plt.subplot2grid(shape=(3, 6), loc=(0, 0), colspan=3),
            plt.subplot2grid((3, 6), loc=(0, 3), colspan=3),
            plt.subplot2grid((3, 6), (1, 0), colspan=3),
            plt.subplot2grid((3, 6), (1, 3), colspan=3),
            plt.subplot2grid((3, 6), (2, 0), colspan=3),
            plt.subplot2grid((3, 6), (2, 3), colspan=3), ]
    # fig, axes = plt.subplots(nrows=3, ncols=col, figsize=(9, 4))
    bplot2 = plot_bp(axes[0], 'DSC', dcs_with_lc)
    # fill with colors
    colors = ['pink', 'lightblue', 'tomato', 'lightgreen', 'hotpink', 'orchid', 'cyan']
    fill_pb_color([bplot2], colors, axes[0])
    hgrid(axes[0], top=1.01, bottom=0, major_gap=.1, minor_gap=0.05, axis='y')
    axes[0].get_xaxis().set_ticks([])
    # =============================================
    # read xls files and fill vectors
    msd_without_lc, msd_with_lc = read_xls(xls_path, results, parent_pth, fields=['msd', 'msd_lc'])
    bplot2 = plot_bp(axes[1], 'MSD', msd_with_lc)
    # fill with colors
    fill_pb_color(([bplot2]), colors, axes[1])
    if test_vali == 0:
        major_gap = 50
    else:
        major_gap = 50
    hgrid(axes[1], top=31, bottom=0, major_gap=10, minor_gap=5, axis='y')
    axes[1].get_xaxis().set_ticks([])
    # =============================================
    # # read xls files and fill vectors
    hd_without_lc, hd_with_lc = read_xls(xls_path, results, parent_pth, fields=['95hd', '95hd_lc'])
    bplot2 = plot_bp(axes[2], '95%HD', hd_with_lc)
    fill_pb_color(([bplot2]), colors, axes[2])
    hgrid(axes[2], top=151, bottom=0, major_gap=50, minor_gap=10, axis='y')
    axes[2].get_xaxis().set_ticks([])
    # =============================================
    cumulative_dice(axes[3], cnn_tags, data=dcs_with_lc, title='Cumulative frequency percentage of dice')
    bp1 = hgrid(axes[3], top=101, bottom=0, major_gap=20, minor_gap=20, axis='y')
    # bp2 = hgrid(axes[1, 3], top=top, bottom=0, major_gap=100, minor_gap=20, axis='y')
    # =============================================
    # read xls files and fill vectors
    bottom_dis, top_dis = read_xls(xls_path, results, parent_pth, fields=['bottom_prep_dis', 'top_prep_dis_lc'])
    bottom_dis = [td * 3 for td in bottom_dis]
    top_dis = [td * 3 for td in top_dis]
    bplot2 = plot_bp(axes[4], 'Top distance', top_dis)
    fill_pb_color(([bplot2]), colors, axes[4])
    hgrid(axes[4], top=31, bottom=-60, major_gap=10, minor_gap=5, axis='y')
    axes[4].get_xaxis().set_ticks([])
    bplot2 = plot_bp(axes[5], 'Bottom distance', bottom_dis)
    fill_pb_color(([bplot2]), colors, axes[5])
    hgrid(axes[5], top=61, bottom=-30, major_gap=10, minor_gap=5, axis='y')
    axes[5].get_xaxis().set_ticks([])
    axes[5].legend(
        [bplot2["boxes"][0], bplot2["boxes"][1], bplot2["boxes"][2], bplot2["boxes"][3], bplot2["boxes"][4],
         # bplot2["boxes"][5],
         # bplot2["boxes"][6]
         ],
        cnn_tags, loc='lower center',
        bbox_to_anchor=(-.05, -.5),
        fancybox=True, shadow=True, ncol=3)
# fig.subplots_adjust(bottom=0.3)  # or whatever

    plt.show()
    print(2)
    # adding horizontal grid lines

