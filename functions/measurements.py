import tensorflow as tf
import SimpleITK as sitk
#import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import datetime

class _measure:
    def __init__(self):
        self.eps=0.00001
        print("measurement create object")

    def dice(self, im1, im2):

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / (im1.sum() + im2.sum()+self.eps)

    def jaccard(self, im1, im2):

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Jaccard coefficient
        intersection = np.logical_and(im1, im2)

        return intersection.sum() / (im1.sum() + im2.sum() - intersection.sum()+self.eps)
    def compute_dice_jaccard(self,res,gt):
        im1 = np.asarray(res).astype(np.bool)
        im2 = np.asarray(gt).astype(np.bool)
        d=(self.dice(im1, im2))
        j=(self.jaccard(im1, im2))
        return d,j
    def plot_diagrams(self,d,j,name,labels,path):
        # fig = plt.figure()
        # l=len(d)
        # line1, =plt.plot(range(l), d,  'ro',label="Dice")#'bo--',range(2), j, 'rs--')
        # line2, =plt.plot(range(l), j,  'bs',label="Jaccard")#,range(2), j, 'rs--')
        # # first_legend = plt.legend(handles=[line1], loc=1)
        # ax = plt.gca().add_artist(first_legend)

        n_groups = len(j)
        means_frank = d
        means_guido = j

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 1

        rects1 = plt.bar(index, d, bar_width,
                         alpha=opacity,
                         color='b',
                         label='Dice')
        #
        rects2 = plt.bar(index + bar_width, j, bar_width,
                         alpha=opacity,
                         color='r',
                         label='Jaccard')
        ax.set_yticks(np.arange(0, 1.1, 0.2))

        plt.xlabel('Slices')
        plt.ylabel('Accuracy')
        title=name+': '+'jaccard: (%.2f,%.2f), dice: (%.2f,%.2f)' % (min(j), max(j), min(d), max(d))

        plt.title(title)
        # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
        plt.legend( loc=4)

        if len(labels):
            # ax.set_xticklabels(labels=labels)
            if len(labels):
                plt.xticks(index + bar_width / 2, labels)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(90)

        plt.tight_layout()
        # plt.show()

        # fig = plt.figure()
        # l = tuple(range(len(d)))
        # v = np.arange(len(l))
        # line1, = plt.bar(v, d, align='center', alpha=0.5)  # 'bo--',range(2), j, 'rs--')

        # plt.legend(handles=[line1,line2], loc=4)
        # plt.xlabel('Slice')
        # plt.ylabel('Accuracy')
        # plt.title('jaccard: (%.2f,%.2f), dice: (%.2f,%.2f)' % (min(j), max(j), min(d), max(d)))
        # plt.show()
        fig.savefig(path+'/dice_jaccard_'+name+'.png')
