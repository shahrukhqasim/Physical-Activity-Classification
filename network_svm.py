from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def get_x_data(D):
    A=D.reshape(-1,12040);
    return A

def ber(x,y):
    A = confusion_matrix(x, y)
    N = np.shape(A)[0]
    X = 0
    for i in range(N):
        S=np.sum(A[i,:]);
        X=(S+A[i,i])/S
    return X/N



def run_svm(X,Y,X_t,Y_t):
    X=get_x_data(X)
    X_t=get_x_data(X_t)

    clf = svm.NuSVC(kernel='rbf',nu=0.01)
    clf.fit(X, Y)

    pp=clf.predict(X_t)

    return sum(pp==Y_t)/np.shape(Y_t),ber(Y_t,pp)