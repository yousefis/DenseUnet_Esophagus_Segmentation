import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
#import math as math
import numpy as np

from read_data import _read_data
import random

import SimpleITK as sitk
import matplotlib.pyplot as plt
# import math as math
import numpy as np

from read_data import _read_data

train_tag='train/'
validation_tag='validation/'
test_tag='Esophagus/'
# img_name='CTpadded.mha'
# label_name='GTV_CTpadded.mha'
# torso_tag='Torsopadded.mha'

img_name = 'CT_padded.mha'
label_name = 'GTV_prim_padded.mha'
torso_tag = 'CTpadded.mha'
data=2

def calculateJac(DVF, voxelSize=[1, 1, 3]):
    '''
    :param DVF: a numpy array with shape of (sizeY, sizeX, 2) or (sizeZ, sizeY, sizeX, 3). You might use np.transpose before this function to correct the order of DVF shape.
    :param voxelSize: physical voxel spacing in mm
    :return: Jac
    '''

    if (len(np.shape(DVF)) - 1) != len(voxelSize):
        raise ValueError ('dimension of DVF is {} but dimension of voxelSize is {}'.format(
            len(np.shape(DVF)) - 1, len(voxelSize)))
    T = np.zeros(np.shape(DVF), dtype=np.float32) # derivative should be calculated on T which is DVF + indices (world coordinate)
    indices = [None] * (len(np.shape(DVF)) - 1)
    DVF_grad = []

    if len(voxelSize) == 2:
        indices[0], indices[1] = np.meshgrid(np.arange(0, np.shape(DVF)[0]), np.arange(0, np.shape(DVF)[1]), indexing='ij')
    if len(voxelSize) == 3:
        indices[0], indices[1], indices[2] = np.meshgrid(np.arange(0, np.shape(DVF)[0]),
                                                         np.arange(0, np.shape(DVF)[1]),
                                                         np.arange(0, np.shape(DVF)[2]), indexing='ij')

    for d in range(len(voxelSize)):
        indices[d] = indices[d] * voxelSize[d]
        T[:, :, :, d] = DVF[:, :, :, d] + indices[d]
        DVF_grad.append(np.gradient(T[:, :, :, d]))  # DVF.grad can be calculated in one shot without for loop.
    if len(voxelSize) == 2:
        Jac = DVF_grad[0][0] * DVF_grad[1][1] - DVF_grad[0][1] * DVF_grad[1][0]
        #       f0/dir0      *   f1/dir1      -    f0/dir1     *   f1/dir0

    if len(voxelSize) == 3:
        Jac = (DVF_grad[0][0] * DVF_grad[1][1] * DVF_grad[2][2] +  # f0/dir0 + f1/dir1 + f2/dir2
               DVF_grad[0][1] * DVF_grad[1][2] * DVF_grad[2][0] +  # f0/dir1 + f1/dir2 + f2/dir0
               DVF_grad[0][2] * DVF_grad[1][0] * DVF_grad[2][1] -
               DVF_grad[0][2] * DVF_grad[1][1] * DVF_grad[2][0] -
               DVF_grad[0][1] * DVF_grad[1][0] * DVF_grad[2][2] -
               DVF_grad[0][0] * DVF_grad[1][2] * DVF_grad[2][1]
               )

    return Jac

def Bspline_distort( CT_image, GTV_image, Torso_image, Penalize_image,displace_range):
    # random.seed(100)
    grid_space = 9

    # displace_range = list(itertools.islice(gen, 1))[0]

    # displace_range = random.randint(5, 20)

    spacing = CT_image.GetSpacing()
    origin = CT_image.GetOrigin()
    direction = CT_image.GetDirection()

    # define transform:
    BCoeff = sitk.BSplineTransformInitializer(CT_image, [grid_space, grid_space, grid_space], order=3)
    # The third parameter for the BSplineTransformInitializer is the spline order It defaults to 3

    displacements = np.random.uniform(-displace_range, displace_range, int(len(BCoeff.GetParameters()) ))
    param_no = np.int(np.ceil(np.power(len(displacements) / 3, 1 / 3)))

    # Xdisplacements = scipy.ndimage.filters.gaussian_filter(
    #     np.reshape(displacements[0: param_no * param_no * param_no],
    #                [param_no, param_no, param_no]), 1.5)
    # Ydisplacements = scipy.ndimage.filters.gaussian_filter(
    #     np.reshape(displacements[param_no * param_no * param_no: 2 * param_no * param_no * param_no],
    #                [param_no, param_no, param_no]), 1.5)
    # Zdisplacements = scipy.ndimage.filters.gaussian_filter(
    #     np.reshape(displacements[2 * param_no * param_no * param_no:3 * param_no * param_no * param_no],
    #                [param_no, param_no, param_no]), 1.5)

    # displacements = np.hstack((np.reshape(Xdisplacements, -1),
    #                            np.reshape(Ydisplacements, -1),
    #                            np.reshape(Zdisplacements, -1)))

    BCoeff.SetParameters(displacements)

    DVF_filter = sitk.TransformToDisplacementFieldFilter()
    DVF_filter.SetSize(CT_image.GetSize())
    DVF_filter.SetOutputOrigin(CT_image.GetOrigin())
    # DVF_filter.SetOrigin(CT_image.GetOrigin())
    DVF_ = DVF_filter.Execute(BCoeff)
    jac=calculateJac(sitk.GetArrayFromImage(DVF_))
    plt.figure()
    plt.hist(np.ravel(jac))
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(DVF_)[100,:,:,0])
    plt.figure()
    plt.hist(displacements)
    plt.figure()
    plt.hist(np.ravel(sitk.GetArrayFromImage(DVF_)))
    # define sampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(BCoeff)
    resampler.SetReferenceImage(CT_image)  # set input image
    resampler.SetInterpolator(sitk.sitkBSpline)  # set interpolation method
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetDefaultPixelValue(-1000)

    CT_deformed = sitk.Resample(CT_image, BCoeff)

    # define sampler for gtv
    bth = sitk.BinaryThresholdImageFilter()
    GTV_image = bth.Execute(GTV_image, 255, 255, 1, 0)
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(BCoeff)
    resampler.SetReferenceImage(GTV_image)  # set input image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # set interpolation method
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    GTV_deformed = sitk.Resample(GTV_image, BCoeff)

    # define sampler for penalize
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(BCoeff)
    resampler.SetReferenceImage(Penalize_image)  # set input image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # set interpolation method
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    Penalize_deformed = sitk.Resample(Penalize_image, BCoeff)

    # define sampler for torso
    # bth = sitk.BinaryThresholdImageFilter()
    # Torso_image = bth.Execute(Torso_image, 255, 255, 1, 0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(BCoeff)
    resampler.SetReferenceImage(Torso_image)  # set input image
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # set interpolation method
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    Torso_deformed = sitk.Resample(Torso_image, BCoeff)
    return CT_deformed, GTV_deformed, Torso_deformed, Penalize_deformed,DVF_


def random_gen(low, high):
    while True:
        yield random.randrange(low, high)

_rd = _read_data(data=data,train_tag=train_tag, validation_tag=validation_tag, test_tag=test_tag,
                             img_name=img_name, label_name=label_name)


# test_CTs, test_GTVs ,test_Torsos= _rd.read_imape_path(test_path)

train_CTs, train_GTVs, train_Torso, train_penalize, \
validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
test_CTs, test_GTVs, test_Torso, test_penalize = _rd.read_data_path()
for img_index in range(len(train_CTs)):
    CT_image1 = sitk.ReadImage(''.join(train_CTs[int(img_index)]))
    voxel_size = CT_image1.GetSpacing()
    origin = CT_image1.GetOrigin()
    direction = CT_image1.GetDirection()
    GTV_image1 = sitk.ReadImage(''.join(train_GTVs[int(img_index)]))
    Torso_image1 = sitk.ReadImage(''.join(train_Torso[int(img_index)]))
    Penalize_image1 = sitk.ReadImage(''.join(train_penalize[int(img_index)]))
    for i in range(6):
        [CT_image1,GTV_image1,Torso_image1,Penalize_image1,DVF_]=Bspline_distort(CT_image1,GTV_image1,Torso_image1,Penalize_image1,displace_range=i+1)


