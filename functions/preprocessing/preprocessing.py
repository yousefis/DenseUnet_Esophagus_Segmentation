import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import SimpleITK as sitk
from functions.data_reader.read_data3 import _read_data
class surface_map:
    def image_padding(self, img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img

    def surface_map(self,gt):
        Gt = sitk.ReadImage(gt)


        # dilation
        radius = 5
        DilateFilter = sitk.BinaryDilateImageFilter()
        DilateFilter.SetKernelRadius(radius)
        DilateFilter.SetForegroundValue(1)
        dilated5 = DilateFilter.Execute(Gt)

        # # erosion
        # radius = 5
        ErodeFilter = sitk.BinaryErodeImageFilter()
        # ErodeFilter.SetKernelRadius(radius)
        # ErodeFilter.SetForegroundValue(1)
        # eroded5 = ErodeFilter.Execute(Gt)

        radius = 1
        ErodeFilter.SetKernelRadius(radius)
        ErodeFilter.SetForegroundValue(1)
        eroded1 = ErodeFilter.Execute(Gt)

        res = (dilated5 ) - (Gt - eroded1)
        nnm=str.replace(gt,'GTV','SURD')
        sitk.WriteImage(res,nnm)
        print(nnm)

        # return res

if __name__=='__main__':
    sm = surface_map()

    _rd = _read_data(data=2, train_tag='', validation_tag='',
                     test_tag='',
                     img_name='', label_name='', torso_tag='')
    train_CTs, train_GTVs, train_Torso, train_penalize, \
    validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
    test_CTs, test_GTVs, test_Torso, test_penalize = _rd.read_data_path(fold=0)
    gtvs = np.append(train_GTVs,np.append(validation_GTVs,test_GTVs))
    num_cores = multiprocessing.cpu_count()

    # for i in range(8):
    #     I = sitk.ReadImage(str.replace(gtvs[i], '_pad87', ''))
    #     padd_zero = 87
    #     padded_img = sm.image_padding(img=I,
    #                                     padLowerBound=[int(padd_zero ) + 1, int(padd_zero ) + 1,
    #                                                    int(padd_zero ) + 1],
    #                                     padUpperBound=[int(padd_zero ), int(padd_zero ), int(padd_zero )],
    #                                     constant=0)
    #     padded_img.SetSpacing(I.GetSpacing())
    #     padded_img.SetOrigin(I.GetOrigin())
    #     padded_img.SetDirection(I.GetDirection())
    #     sitk.WriteImage(padded_img, gtvs[i])

    print(num_cores)
    Parallel(n_jobs=num_cores)(
        delayed(sm.surface_map)(gt=gtvs[i]
                                     )
        for i in range(len(gtvs)))  #