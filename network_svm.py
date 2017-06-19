from sklearn import svm
import numpy as np

def get_x_data(D):
    A=D.reshape(-1,12040);
    return A

def run_svm(X,Y,X_t,Y_t):
    clf = svm.NuSVC(kernel='sigmoid',nu=0.01,probability=True)
    # print(Y)
    print("Hello world")
    clf.fit(get_x_data(X), Y)
    pp=clf.predict(get_x_data(X_t))
    print(np.shape(pp))
    print(np.shape(Y_t))
    print(sum(pp==Y_t)/np.shape(Y_t))
    return sum(pp==Y_t)/np.shape(Y_t)