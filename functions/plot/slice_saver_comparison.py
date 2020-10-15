import  SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cv2

def get_all_boundary_edges(bool_img):
    """
    Get a list of all edges
    (where the value changes from 'True' to 'False') in the 2D image.
    Return the list as indices of the image.
    """
    ij_boundary = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            ij_boundary.append(np.array([[i, j+1],
                                         [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            ij_boundary.append(np.array([[i+1, j],
                                         [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            ij_boundary.append(np.array([[i, j],
                                         [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            ij_boundary.append(np.array([[i, j],
                                         [i, j+1]]))
    if not ij_boundary:
        return np.zeros((0, 2, 2))
    else:
        return np.array(ij_boundary)




def close_loop_boundary_edges(xy_boundary, clean=True):
    """
    Connect all edges defined by 'xy_boundary' to closed
    boundary lines.
    If not all edges are part of one surface return a list of closed
    boundaries is returned (one for every object).
    """

    boundary_loop_list = []
    while xy_boundary.size != 0:
        # Current loop
        xy_cl = [xy_boundary[0, 0], xy_boundary[0, 1]]  # Start with first edge
        xy_boundary = np.delete(xy_boundary, 0, axis=0)

        while xy_boundary.size != 0:
            # Get next boundary edge (edge with common node)
            ij = np.nonzero((xy_boundary == xy_cl[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                xy_cl.append(xy_cl[0])
                break

            xy_cl.append(xy_boundary[i, (j + 1) % 2, :])
            xy_boundary = np.delete(xy_boundary, i, axis=0)

        xy_cl = np.array(xy_cl)

        boundary_loop_list.append(xy_cl)

    return boundary_loop_list

def plot_world_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ij_boundary = get_all_boundary_edges(bool_img=bool_img)
    xy_boundary = ij_boundary - 0.5
    xy_boundary = close_loop_boundary_edges(xy_boundary=xy_boundary)
    cl = LineCollection(xy_boundary, **kwargs)
    ax.add_collection(cl)
def hist_equalizer(img):
    # equ = cv2.equalizeHist(img)
    # res = np.hstack((img, equ))  # stacking images side-by-side
    # plt.imshow(res)
    # img = I[slice_no, :, :]
    hist, bins = np.histogram(img.flatten(), np.max(img) - np.min(img), [np.min(img), np.max(img)])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * cdf_m.max() / (1000 - 100)
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    plt.imshow(img2,cmap='gray')
    plt.show()

def dice( im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    print('intersect:%d, sum1:%d, sum2:%d'%(intersection.sum(),im1.sum() , im2.sum()))
    d=2. * intersection.sum() / (im1.sum() + im2.sum() + .0000001)

    return d

parent_path="/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/"
path=['33533_0.75_4-train1-04172020_140/',#DUnet
      '33533_0.75_4-train1-05082020_090/',  #dice  Nodistancemap
      '33533_0.75_4-train1-07052020_000/',  #dice+attention channel no distancemap
      '33533_0.75_4-train1-07142020_020/',  #dice+attention spatial  no distancemap
      '33533_0.75_4-train1-07102020_140/',  # dice+attention channel+spatial  no distancemap
      '33533_0.75_4-train1-08242020_1950240/' , # dice+attention spatial  no distancemap +channel skip att+bourndry loss     
      ]
tags=['DUnet','DDUnet','DDCAUnet','DDSAUnet','DDSCAUnet','DDSAUnetSkipA']
images=[
    # "RLZU_2013-07-23_4DCT_80"
    # "RJSC_2013-07-09_4DCT_40",
    # "TEST080_2016-09-06",
    # "TEST103_2018-01-11",
    # "zz2408044448_zz2408044448",
    # "zz3710236225_zz3710236225",
    # "zz3744705519_zz3744705519"
    # "TEST091_2017-03-13",
    # "TEST086_2016-12-21",
    # "TEST108_2017-09-18",
    # "zz2816682167_zz2816682167",
    # "zz3466954516_zz3466954516",
    # "zz197056916_zz197056916",
    # "RGLA_2013-06-11_4DCT_80"
        ]
image="zz1797328816_zz1797328816"

slice_no=[[113,120, ]]


res_vali='result_vali/'
save_path = '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/journal3/'
for p,c,k in zip(path,range(len(tags)) ,range(1,8)) :
    img_nm = image + "_ct.mha"
    img_gtv = image  + "_gtv.mha"
    img_res = image  + "_result_lc.mha"
    II = sitk.ReadImage(parent_path + p + res_vali + img_nm)
    I = sitk.GetArrayFromImage(II)
    gtv = sitk.GetArrayFromImage(sitk.ReadImage(parent_path + p + res_vali + img_gtv))
    slice_no = 118#np.unique(np.where(gtv)[0])
    j=slice_no
    res=sitk.GetArrayFromImage(sitk.ReadImage(parent_path+p+res_vali+img_res))

    dsc=dice( gtv[j,:,:], res[j,:,:])
    print(dsc)
    a=160
    b=-150
    # plt.figure(frameon=False)
    # plt.imshow(I[j,:,:], cmap='gray')
    # plt.axis('off')
    sitk_img = sitk.GetImageFromArray(np.expand_dims(I[j,:,:],0))
    sitk_img.SetDirection(direction=II.GetDirection())
    sitk_img.SetOrigin(origin=II.GetOrigin())
    sitk_img.SetSpacing(spacing=II.GetSpacing())
    sitk.WriteImage(sitk_img,save_path+str(j)+'_'+img_nm.rsplit('_ct.mha')[0]+'_'+str(j)+'.mha')

    sitk_img = sitk.GetImageFromArray(np.expand_dims(gtv[j,:,:],0))
    sitk_img.SetDirection(direction=II.GetDirection())
    sitk_img.SetOrigin(origin=II.GetOrigin())
    sitk_img.SetSpacing(spacing=II.GetSpacing())
    sitk.WriteImage(sitk_img,save_path+str(j)+'_'+img_nm.rsplit('_ct.mha')[0]+'_'+str(j)+'_gtv.mha')

    sitk_img = sitk.GetImageFromArray(np.expand_dims(res[j,:,:],0))
    sitk_img.SetDirection(direction=II.GetDirection())
    sitk_img.SetOrigin(origin=II.GetOrigin())
    sitk_img.SetSpacing(spacing=II.GetSpacing())
    sitk.WriteImage(sitk_img,save_path+str(j)+'_'+str(k)+'_'+tags[c]+'_'+img_nm.rsplit('_ct.mha')[0]+'_'+str(j)+'_res_'+str(dsc)+'.mha')

    # plt.savefig(save_path+img_nm.rsplit('_ct.mha')[0]+'_'+str(slice_no)+'.png', bbox_inches='tight', pad_inches=0,dpi=500)
    # plot_world_outlines(gtv[slice_no,:,:].T, lw=1, color='springgreen')
    # plot_world_outlines(res[slice_no,:,:].T, lw=1, color='r')
    # plt.axis('off')



        # plt.savefig(save_path+str(c)+'_'+img_nm.rsplit('_ct.mha')[0]+'_'+str(j)+'_'+str(dsc)+'.png', bbox_inches='tight', pad_inches=0,dpi=500)

