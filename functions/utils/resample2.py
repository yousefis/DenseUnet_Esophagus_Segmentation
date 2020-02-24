from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
from read_data import _read_data
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------------------------------
#run this code on windows for extracting zero padded tensor and images
# --------------------------------------------------------------------------------------------------------
def zeropadding(name,pad_size,value,shift=0):

    CT_image1 = sitk.ReadImage(name)
    original_direction = CT_image1.GetDirection()
    original_origin = CT_image1.GetOrigin()

    original_spacing = CT_image1.GetSpacing()


    tag = '_padded'
    CT_image1 = sitk.GetArrayFromImage(CT_image1)
    if shift==1:
        CT_image2 = (np.array(CT_image1[:, :, :]) - np.ones(shape=CT_image1.shape) * 1024)
        CT_image2[np.where(CT_image2 > 1500)] = -1000
    else:
        CT_image2=CT_image1
    pad_size=pad_size*2
    CT_image2=np.pad(CT_image2, ((int(pad_size / 2) + 1, int(pad_size / 2)), (int(pad_size / 2)+ 1, int(pad_size / 2)),
                   (int(pad_size / 2)+ 1, int(pad_size / 2))),
           mode='constant', constant_values=value)

    CT_image1 = sitk.GetImageFromArray(CT_image2.astype(np.short))

    CT_image1.SetDirection(direction=original_direction)
    CT_image1.SetOrigin(origin=original_origin)
    CT_image1.SetSpacing(spacing=original_spacing)


    sitk_isotropic_xslice = CT_image1

    resample_name = name.split('.mha')[0] + tag + '.mha'
    sitk.WriteImage(sitk_isotropic_xslice, resample_name)
    print(resample_name)

def padding_images(CTs,GTV,Torso,pad_size):

    for i in range(len(CTs)):
        zeropadding(CTs[i], pad_size,value=-1000,shift=1)
        zeropadding(GTV[i], pad_size,value=0)
        # zeropadding(Torso[i], pad_size,value=0)

def fun_resample(new_spacing,name,min_normal,max_normal,isnormal,gtv, shift_need):
    tag=''

    if len(new_spacing):
        tag = '_re'+''.join(map(str, new_spacing))
    CT_image1 = sitk.ReadImage(name)
    original_direction = CT_image1.GetDirection()
    original_origin = CT_image1.GetOrigin()

    original_spacing = CT_image1.GetSpacing()
    original_size = CT_image1.GetSize()

    min_spacing = min(CT_image1.GetSpacing())
    if (shift_need==1):
        tag = '_shifted'
        CT_image1 = sitk.GetArrayFromImage(CT_image1)
        # CT_image1-1000
        CT_image2 =(np.array(CT_image1[:, :, :]) - np.ones(shape=CT_image1.shape) * 1024)
        CT_image2[np.where(CT_image2>1500)]=-1000

        CT_image1 = sitk.GetImageFromArray(CT_image2.astype(np.short))

        CT_image1.SetDirection(direction=original_direction)
        CT_image1.SetOrigin(origin=original_origin)
        CT_image1.SetSpacing(spacing=original_spacing)



    if (isnormal==1) & (gtv!=1):
        CT_image1=sitk.GetArrayFromImage(CT_image1)
        CT_image1= return_normal_image(CT_image1, 0, 1,min_normal,max_normal)
        CT_image1 =sitk.GetImageFromArray(CT_image1.astype(np.float32))
        tag+='_normal'
    if (isnormal == 1) & (gtv == 1):
        tag += '_normal'
    if gtv==1:
        CT_image1 = sitk.GetArrayFromImage(CT_image1)
        CT_image1=np.int32(CT_image1/ np.max(CT_image1))
        CT_image1 = sitk.GetImageFromArray(CT_image1.astype(np.uint8))

    if len(new_spacing):
        new_size = [int(round(original_size[0]*(original_spacing[0]/new_spacing[0]))),
                    int(round(original_size[1]*(original_spacing[1]/new_spacing[1]))),
                    int(round(original_size[2]*(original_spacing[2]/new_spacing[2])))]
        resampleSliceFilter = sitk.ResampleImageFilter()

        sitk_isotropic_xslice = resampleSliceFilter.Execute(CT_image1, new_size, sitk.Transform(),
                                                            sitk.sitkNearestNeighbor, CT_image1.GetOrigin(),
                                                            new_spacing, CT_image1.GetDirection(), 0,
                                                            CT_image1.GetPixelID())
    else:
        sitk_isotropic_xslice=CT_image1



    resample_name = name.split('.mha')[0] + tag + '.mha'
    sitk.WriteImage(sitk_isotropic_xslice,resample_name)
    print(resample_name)

def resample_images(CTs,GTV,new_spacing,min_normal,max_normal,isnormal,shift_need):

    for i in range(len(CTs)):
        fun_resample(new_spacing,''.join(CTs[int(i)]),min_normal,max_normal,isnormal,gtv=0,shift_need=shift_need)
        fun_resample( new_spacing,''.join(GTV[int(i)]),min_normal,max_normal,isnormal,gtv=1,shift_need=shift_need)


def torso_mul(Torsos):
    for img_index in range(len(Torsos)):


        Torso_image = sitk.ReadImage(''.join(Torsos[int(img_index)]))

        original_direction = Torso_image.GetDirection()
        original_origin = Torso_image.GetOrigin()
        original_spacing = Torso_image.GetSpacing()

        Torso_image = sitk.GetArrayFromImage(Torso_image)
        Torso_image_mul=creat_mask(Torso_image.shape) * Torso_image

        CT_image1 = sitk.GetImageFromArray(Torso_image_mul.astype(np.uint8))

        CT_image1.SetDirection(direction=original_direction)
        CT_image1.SetOrigin(origin=original_origin)
        CT_image1.SetSpacing(spacing=original_spacing)

        resample_name = Torsos[int(img_index)].split('.mha')[0] + '_mul_mask' + '.mha'
        sitk.WriteImage(CT_image1, resample_name)


def return_normal_image(CT_image,min_range,max_range,min_normal,max_normal):
    return (max_range - min_range) * (
    (CT_image - min_normal) / (max_normal - min_normal)) + min_range

def creat_mask( shape):
    patch_window=47
    torso_mask = np.ones((shape[0] - int(patch_window),
                          shape[1] - int(patch_window),
                          shape[2] - int(patch_window)))

    torso_mask = np.pad(torso_mask, (
        (int(patch_window / 2) + 1, int(patch_window / 2)),
        (int(patch_window / 2) + 1, int(patch_window / 2)),
        (int(patch_window / 2) + 1, int(patch_window / 2)),
    ),
                        mode='constant', constant_values=0)
    return torso_mask


def get_torso_mask(ptpulmo, original_image_filename, torso_dir, pulmo_setting_file,rename):
    print("run ptpulmo")

    # original_image_filename='C:\\Users\\syousefi1\\Desktop\\CTpadded.mha'
    # torso_dir='C:\\Users\\syousefi1\\Desktop\\'

    cmdStr = "%s -in %s -ps %s -out %s -torso" % (ptpulmo, original_image_filename, pulmo_setting_file, torso_dir)
    print(cmdStr)
    if not os.system(cmdStr) == 0:
        print("ptpulmo failed")
        sys.exit(1)
    os.rename(filename, filename[7:])
def extract_torsos(CTs):
    excutablePath = os.path.join('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Torso/')
    ptpulmo = os.path.join(excutablePath, "ptpulmo.exe")
    pulmo_setting_file = os.path.join(excutablePath, "PulmoDefaultSettings.psf")
    # extract torsos:
    for i in range(0, len(CTs)):
        splitted_patient_name = CTs[i].split('/')
        torso_dir = '/'.join(splitted_patient_name[0:10])+'/'
        rename=splitted_patient_name[10].split('.mha')[0]+'_Torso.mha'
        # if not os.path.exists(torso_dir):
        #     os.mkdir(torso_dir)

        get_torso_mask(ptpulmo, CTs[i], torso_dir, pulmo_setting_file,rename)
def resample_all(new_spacing,isnormal,shift_need):
    # _rd = _read_data(2,train_tag='train/',validation_tag='validation/',test_tag='test/',
    #              img_name='CT.mha',label_name='GTV_CT.mha')
    # _rd = _read_data(1, train_tag='prostate_train/', validation_tag='prostate_validation/', test_tag='prostate_test/',
    #                  img_name='CTImage.mha', label_name='Bladder.mha')
    _rd = _read_data(2, train_tag='train/', validation_tag='validation/', test_tag='Extra/',
                     img_name='CT.mha', label_name='GTV_prim.mha')

    '''read path of the images for train, test, and validation'''

    train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
    test_CTs, test_GTVs, test_Torso, depth, width, height = _rd.read_image_path()

    # extract_torsos(train_CTs)

    # padding and casting:
    # padding_images(train_CTs, train_GTVs,train_Torso, 57)
    # padding_images(validation_CTs, validation_GTVs, validation_Torso, 57)
    padding_images(test_CTs, test_GTVs, test_Torso, 57)
    # then transfer data to windows in order to extract torso files



resample_all(new_spacing = [],isnormal=0, shift_need=1)
