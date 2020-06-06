from functions.data_reader.read_data3 import _read_data
import SimpleITK as sitk
import numpy
if __name__=='__main__':
    data=2
    _rd = _read_data(data=data, train_tag='', validation_tag='',
                     test_tag='',
                     img_name='', label_name='', torso_tag='')
  
    

    '''read path of the images for train, test, and validation'''

    train_CTs, train_GTVs, train_Torso, train_penalize, \
    validation_CTs, validation_GTVs, validation_Torso, validation_penalize, \
    test_CTs, test_GTVs, test_Torso, test_penalize = _rd.read_data_path(fold=0)

    for i in range(len(train_GTVs)):
        I=sitk.GetArrayFromImage(sitk.ReadImage(train_GTVs[i]))
        len(np.where(I)[0])
