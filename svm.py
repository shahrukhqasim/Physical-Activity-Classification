import numpy as np
import network_svm

labels_file = 'labels.npy'
data_file = 'data.npy'

labels = np.load(labels_file)
D = np.load(data_file)

svm=np.zeros((10));
svmb=np.zeros((10));

for i in range(10):
    indices = np.random.permutation(D.shape[0])
    ind_1 = indices[0:32]
    ind_2 = indices[32:64]
    ind_3 = indices[64:96]
    ind_4 = indices[96:129]

    set_1_train = np.concatenate((ind_1, ind_2, ind_3))
    set_1_test = ind_4

    set_2_train = np.concatenate((ind_1, ind_2, ind_4))
    set_2_test = ind_3

    set_3_train = np.concatenate((ind_1, ind_3, ind_4))
    set_3_test = ind_2

    set_4_train = np.concatenate((ind_2, ind_3, ind_4))
    set_4_test = ind_1

    X_1 = D[set_1_train, :, :]
    Y_1 = labels[set_1_train]
    X_t_1 = D[set_1_test, :, :]
    Y_t_1 = labels[set_1_test]

    X_2 = D[set_2_train, :, :]
    Y_2 = labels[set_2_train]
    X_t_2 = D[set_2_test, :, :]
    Y_t_2 = labels[set_2_test]

    X_3 = D[set_3_train, :, :]
    Y_3 = labels[set_3_train]
    X_t_3 = D[set_3_test, :, :]
    Y_t_3 = labels[set_3_test]

    X_4 = D[set_4_train, :, :]
    Y_4 = labels[set_4_train]
    X_t_4 = D[set_4_test, :, :]
    Y_t_4 = labels[set_4_test]

    a1,b1=network_svm.run_svm(X_1,Y_1,X_t_1,Y_t_1)
    a2,b2=network_svm.run_svm(X_2,Y_2,X_t_2,Y_t_2)
    a3,b3=network_svm.run_svm(X_3,Y_3,X_t_3,Y_t_3)
    a4,b4=network_svm.run_svm(X_4,Y_4,X_t_4,Y_t_4)

    svm[i]=(a1+a2+a3+a4)/4
    svmb[i]=(b1+b2+b3+b4)/4

    print("Average k fold accuracy is ", svm[i])



print("Average accuracy overall is ", np.average(svm))
print("Average accuracy BER is ", np.average(svmb))