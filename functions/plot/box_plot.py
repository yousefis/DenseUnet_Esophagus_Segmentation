import matplotlib.pyplot as plt
import numpy as np

# fake up some data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 50
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# data = np.concatenate((spread, center, flier_high, flier_low), 0)

collectn_acc=[]
collectn_loss=[]
collectn_all_train_acc=[]
collectn_all_test_acc=[]
collectn_all_train_loss=[]
collectn_all_test_loss=[]
with open('./accuracy/train_unet.txt', "r") as train,\
        open('./accuracy/validation_unet.txt', "r") as test:
    i=1
    len1=100
    train_acc_sum=0
    train_loss_sum=0
    for line in train:
        if 'epoch' not in line:
            [train_acc,train_loss]=line.split(", ")
            train_acc_sum=train_acc_sum+float(train_acc)
            train_loss_sum=train_loss_sum+float(train_loss)
            if i%len1==0:
                train_acc_ave=train_acc_sum/len1
                train_loss_ave=train_loss_sum/len1
                collectn_acc.append(float(train_acc_ave))
                collectn_loss.append(float(train_loss_ave))

        else:
            if len(collectn_acc)!=0:
                collectn_all_train_acc.append((collectn_acc))
                collectn_all_train_loss.append((collectn_loss))
        i=i+1
    # i=1
    # test_acc_sum=0
    # test_loss_sum=0
    # for line in test:
    #     if 'epoch' not in line:
    #         [test_acc, test_loss] = line.split(", ")
    #         test_acc_sum = test_acc_sum + test_acc
    #         test_loss_sum = test_loss_sum + test_loss
    #         if i % len == 0:
    #             test_acc_ave = test_acc_sum / len
    #             test_loss_ave = test_loss_sum / len
    #             collectn_acc.append(float(test_acc_ave))
    #             collectn_loss.append(float(test_loss_ave))
    #     else:
    #         if len(collectn_acc) != 0:
    #             collectn_all_test_acc.append(collectn_acc)
    #             collectn_all_test_loss.append(collectn_loss)
y = np.arange(35).reshape(5,7)
print(y)
print(y[1:5:2,::3])
# plt.figure()
# plt.boxplot(collectn_all_train_acc)
# np.reshape(collectn_all_train_acc[::10],(100,220))


# plt.figure()
# plt.boxplot(collectn_all_train_acc[::10][0:10])
# don't show outlier points
plt.figure()
plt.boxplot(collectn_all_train_acc[::10], 0, '')

# horizontal boxes
# plt.figure()
# plt.boxplot(data, 0, 'rs', 0)
#
# # change whisker length
# plt.figure()
# plt.boxplot(data, 0, 'rs', 0, 0.75)
#
# # fake up some more data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 40
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
# data.shape = (-1, 1)
# d2.shape = (-1, 1)
# # data = concatenate( (data, d2), 1 )
# # Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# # This is actually more efficient because boxplot converts
# # a 2-D array into a list of vectors internally anyway.
# data = [data, d2, d2[::2, 0]]
# # multiple box plots on one figure
# plt.figure()
# plt.boxplot(data)
#
plt.show()
print('l')
