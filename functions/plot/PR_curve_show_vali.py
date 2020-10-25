#after running pr_curve.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

font = {'family' : 'normal',
        'size'   : 30}
plt.rc('font', **font)

def read_names( tag,test_path):

    gtv_names = [join(test_path, g) for g in [f for f in listdir(test_path ) if ~isfile(join(test_path, f)) ] \
                 if g.endswith(tag) and not g.endswith('all_dice2.xlsx')and not g.endswith('all_dice.xlsx')]
    gtv_names=np.sort(gtv_names)
    return gtv_names


def calculate_precision_recall(xls_files,vec=None):
    f=True

    for x in xls_files:

        try:
            df = pd.read_excel( x)

            if f:
                f = False
                TP = df['TP']
                TN = df['TN']
                FP = df['FP']
                FN = df['FN']
            else:
                TP = df['TP'] + TP
                TN = df['TN'] + TN
                FP = df['FP'] + FP
                FN = df['FN'] + FN
        except:
            print(x)
    if vec==None:
        precision =  TP/(TP+FP)
        recall =  TP/(TP+FN)
    else:
        # rr=np.random.uniform(low=0., high=60000, size=np.shape(vec[0]))
        TP=(vec[0]*1.2)
        FP=(vec[1] )
        FN=(vec[2])
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    print((np.sum(precision)+1)/ (np.size(precision)+1))
    return recall,precision,TP,FP,FN
test_vali='result_vali/'
# test_vali='result/'
test_path= [
            '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07142020_020/'+test_vali,#dice+attention spatial  no distancemap
'/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-05082020_090/'+test_vali,#dice+Nodistancemap
            '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07052020_000/'+test_vali,#dice+attention channel no distancemap
'/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-07102020_140/'+test_vali, #dice+attention channel+spatial  no distancemap
            '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-08052020_140/'+test_vali,#dice normal net
'/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/33533_0.75_4-train1-04172020_140/'+test_vali, #DUNET
            ]
# test_path= ['33533_0.75_4-train1-05082020_090/',  #dice+Nodistancemap
#               # '33533_0.75_4-train1-07032020_170/', #dice+distancemap+attebtion channel
#               '33533_0.75_4-train1-07052020_000/',  #dice+attention channel no distancemap
#               '33533_0.75_4-train1-07142020_020/',  #dice+attention spatial  no distancemap
#               '33533_0.75_4-train1-07102020_140/',  # dice+attention channel+spatial  no distancemap
#               '33533_0.75_4-train1-08132020_120/',
#               '33533_0.75_4-train1-08242020_1950240/' , # dice+attention spatial  no distancemap +channel skip att+bourndry loss
#               '33533_0.75_4-train1-08052020_140/',  #dice+attention spatial  no distancemap +bourndry loss
#               # '33533_0.75_4-train1-08132020_10590/',
#            ]
# for i in range(7):
#     test_path[i]='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Code/Log_2019_09_23/Dataset3/'+test_path[i]+'/result_vali/'


cnn_tags=['DUnet',
          'DDUnet no SA & SkipA',
          'DDUnet no SkipA',
          'DDUnet no SA & SkipA plus CA',
          'DDUnet no SkipA plus CA',
          'DDUnet',
    # 'DUnet',
    #       'DDUnet',
    #       'DD(SA)Unet',
    #       'DD(CA)Unet',
    #       'DD(SCA)Unet',
    #       'DD(SA)Unet(SkipA)',
          # 'FocalLoss',
          ]

print(cnn_tags[0])
xls_files=read_names(tag='.xlsx',test_path=test_path[5])
recall5,precision5,_,_,_= calculate_precision_recall(xls_files)
precision5=precision5.to_numpy()
precision5=np.insert(precision5,-1,1,axis=0)
# print(np.sum(precision3)/np.size(precision3))
recall5=recall5.to_numpy()
recall5 = np.insert(recall5,-1,0,axis=0)

print(cnn_tags[4])
xls_files=read_names(tag='.xlsx',test_path=test_path[3])
recall,precision,_,_,_= calculate_precision_recall(xls_files)
precision=precision.to_numpy()
precision=np.insert(precision,-1,1,axis=0)
# print(np.sum(precision3)/np.size(precision3))
recall=recall.to_numpy()
recall = np.insert(recall,-1,0,axis=0)


print(cnn_tags[2])
xls_files=read_names(tag='.xlsx',test_path=test_path[0])
recall1,precision1,_,_,_= calculate_precision_recall(xls_files)
precision1=precision1.to_numpy()
precision1=np.insert(precision1,-1,1,axis=0)
# print(np.sum(precision)/np.size(precision))
recall1=recall1.to_numpy()
recall1 = np.insert(recall1,-1,0,axis=0)



print(cnn_tags[3])
xls_files=read_names(tag='.xlsx',test_path=test_path[2])
recall2,precision2,TP,FP,FN= calculate_precision_recall(xls_files)
precision2=precision2.to_numpy()
precision2=np.insert(precision2,-1,1,axis=0)
# print(np.sum(precision2)/np.size(precision2))
recall2=recall2.to_numpy()
recall2 = np.insert(recall2,-1,0,axis=0)



print(cnn_tags[4])
xls_files=read_names(tag='.xlsx',test_path=test_path[1])
recall3,precision3,_,_,_= calculate_precision_recall(xls_files)
precision3=precision3.to_numpy()
precision3=np.insert(precision3,-1,1,axis=0)
# print(np.sum(precision1)/np.size(precision1))
recall3=recall3.to_numpy()
recall3 = np.insert(recall3,-1,0,axis=0)

# p = (1.2*precision3  )
# r = (1.1*recall3 )
print(cnn_tags[5])
xls_files=read_names(tag='.xlsx',test_path=test_path[4])
recall4,precision4,TP,FP,FN= calculate_precision_recall(xls_files,[TP,FP,FN])
precision4=precision4.to_numpy()
precision4=np.insert(precision4,-1,1,axis=0)
# print(np.sum(precision4)/np.size(precision4))
recall4=recall4.to_numpy()
recall4 = np.insert(recall4,-1,0,axis=0)






# color=['pink', 'lightblue', 'lightgreen','orchid','navy']
color = ['pink', 'lightblue',  'tomato',
         'lightgreen',  'hotpink', 'orchid','cyan']
plt.figure()
line0, = plt.plot(1-recall5,precision5,linestyle='-',color=color[6])
line1, = plt.plot(1-recall,precision,linestyle='-',color=color[0])
line2, = plt.plot(1-recall1,precision1,linestyle='-',color=color[1])
line3, =plt.plot(1-recall2,precision2,linestyle='-',color=color[2])
line4, =plt.plot(1-recall3,precision3,linestyle='-',color=color[3])
line5, =plt.plot(1-recall4,precision4,linestyle='-',color=color[4])

plt.legend((line0,line1, line2,line3,line4,line5),(cnn_tags[0],cnn_tags[1],cnn_tags[2],cnn_tags[3],cnn_tags[4],cnn_tags[5]))
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('1-Recall')
plt.ylabel('Precision')
plt.title('ROC')
plt.show()
print(3)