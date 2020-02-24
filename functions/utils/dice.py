"""
_dice.py : Dice coefficient for comparing set similarity.
"""
import SimpleITK as sitk
from scipy.spatial.distance import jaccard
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def dice(im1, im2):

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())
def jaccard(im1,im2):


    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Jaccard coefficient
    intersection = np.logical_and(im1, im2)

    return  intersection.sum() / (im1.sum() + im2.sum()-intersection.sum())

gt = sitk.ReadImage('./out/label.png')
d=[]
j=[]
for i in range(100):
    res = sitk.ReadImage(('./out/result_%d.png' %(i)))
    im1 = np.asarray(res).astype(np.bool)
    im2 = np.asarray(gt).astype(np.bool)

    d.append(dice(im1, im2))
    j.append(jaccard(im1, im2))
    print('dice:%f , jaccard: %f' % (d[i], j[i]))

fig = plt.figure()
line1, =plt.plot(range(100), d,  label="Dice", linestyle='-')#'bo--',range(2), j, 'rs--')
line2, =plt.plot(range(100), j,  label="Jaccard", linestyle='--')#,range(2), j, 'rs--')
# first_legend = plt.legend(handles=[line1], loc=1)
# ax = plt.gca().add_artist(first_legend)
plt.legend(handles=[line1,line2], loc=4)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('jaccard: (%.2f,%.2f), dice: (%.2f,%.2f)' %(min(j),max(j),min(d),max(d)))
plt.show()
fig.savefig('./out/dice_jaccard.png')

# plt.plot(i, d, 'r--', i, j, 'bs')
# plt.show()




