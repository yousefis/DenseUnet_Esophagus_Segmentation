from functions.data_reader.read_data3 import _read_data
import pandas as pd
import SimpleITK as sitk
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def fun(name_img,test_CTs):
    I = sitk.GetArrayFromImage(sitk.ReadImage(name_img))
    Ones = len(np.where(I)[0])
    ss = str(test_CTs).split("/")
    name = (ss[10] + '_' + ss[11] + '_' + ss[12].split('%')[0]).split('_CT')[0]
    # list_nm.append(name)
    # list_vol.append(Ones)
    print(name)
    return name,Ones
if __name__=='__main__':
    data=2
    _rd = _read_data(data=data, train_tag='', validation_tag='',
                     test_tag='',
                     img_name='', label_name='', torso_tag='')
  
    

    '''read path of the images for train, test, and validation'''

    train_CTs, train_GTVs, train_Torso, train_penalize, \
    validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
    test_CTs, test_GTVs, test_Torso, test_penalize = _rd.read_data_path(fold=0)
    list_nm = []
    list_vol = []
    # for i in range(len(train_GTVs)):#
    #     I=sitk.GetArrayFromImage(sitk.ReadImage(train_GTVs[i]))
    #     Ones= len(np.where(I)[0])
    #     ss = str(test_CTs[i]).split("/")
    #     name = (ss[10] + '_' + ss[11] + '_' + ss[12].split('%')[0]).split('_CT')[0]
    #     list_nm.append(name)
    #     list_vol.append(Ones)
    #     print(name)

    num_cores = 1#multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores)(
        delayed(fun)(train_GTVs[i],test_CTs[i])
        for i in range(len(train_GTVs)))  #
    df = pd.DataFrame(res,
                      columns=pd.Index(['name','volume'],
                       name='Genus')).round(2)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/dense_net_3d_segmentation-1-dice-tumor--106/out/volume.xlsx',
                            engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

