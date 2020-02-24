import os
import shutil

path = '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/21data1100data2-v4/'
dirs = os.listdir(path)
for d in dirs:
    # os.mkdir(path+d+'/'+d)
    # shutil.move(path+d+'/GTV_re113.mha',path+d+'/'+d+'/GTV_re113.mha')
    # shutil.move(path+d+'/CT_re113.mha',path+d+'/'+d+'/CT_re113.mha')
    # shutil.move(path+d+'/CT_re113_Torso.mha',path+d+'/'+d+'/CT_re113_Torso.mha')
    # print('s')
    os.rename(path+d+'/'+d+'/CT_re113_Torso.mha',path+d+'/'+d+'/CT_Torso_re113.mha')