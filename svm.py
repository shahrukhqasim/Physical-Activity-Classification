import numpy as np
import network_svm

labels_file = 'labels.npy'
data_file = 'data.npy'

labels = np.load(labels_file)
D = np.load(data_file)

ind_1 = np.array([1,2,9,10,17,18,25,26,3,4,6,11,12,14,19,20,22,27,28,30,5,13,21,8,16,24,39,7,15,23,31])-1
ind_2 = np.array([32,33,40,47,48,54,55,34,35,37,41,43,44,49,51,56,57,59,64,42,50,58,66,46,53,61,69,38,45,52,60])-1
ind_3 = np.array([62,63,70,71,77,83,84,91,65,67,72,74,78,79,81,85,86,88,93,94,73,80,87,95,76,82,90,98,68,75,89,97])-1
ind_4 = np.array([92,99,100,107,108,114,122,123,96,101,102,104,109,111,115,116,118,124,125,127,103,110,117,126,106,113,121,129,105,112,119,120,128])-1

set_1_train = np.concatenate((ind_1,ind_2,ind_3))
set_1_test = ind_4

set_2_train = np.concatenate((ind_1,ind_2,ind_4))
set_2_test = ind_3

set_3_train = np.concatenate((ind_1,ind_3,ind_4))
set_3_test = ind_2

set_4_train = np.concatenate((ind_2,ind_3,ind_4))
set_4_test = ind_1

X_1 = D [set_1_train, :, :]
Y_1 = labels[set_1_train]
X_t_1 = D [set_1_test, :, :]
Y_t_1 = labels[set_1_test]

X_2 = D [set_2_train, :, :]
Y_2 = labels[set_2_train]
X_t_2 = D [set_2_test, :, :]
Y_t_2 = labels[set_2_test]

X_3 = D [set_3_train, :, :]
Y_3 = labels[set_3_train]
X_t_3 = D [set_3_test, :, :]
Y_t_3 = labels[set_3_test]

X_4 = D [set_4_train, :, :]
Y_4 = labels[set_4_train]
X_t_4 = D [set_4_test, :, :]
Y_t_4 = labels[set_4_test]


a1=network_svm.try_svm(X_1,Y_1,X_t_1,Y_t_1)
a2=network_svm.try_svm(X_2,Y_2,X_t_2,Y_t_2)
a3=network_svm.try_svm(X_3,Y_3,X_t_3,Y_t_3)
a4=network_svm.try_svm(X_4,Y_4,X_t_4,Y_t_4)

print("Average k fold accuracy is ", (a1+a2+a3+a4)/4)