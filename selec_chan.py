import os
import sys
import numpy as np

def selec_chan(dataset, fea_name):

    fea_fold = os.path.join(dataset,fea_name)
    imgList = os.listdir(fea_fold)

    data_fea = [];
    for name in imgList:
        path = os.path.join(fea_fold,name)
        X = np.load(path)
        chan_num = X.shape[0]
        X_Sum = np.reshape(X,(chan_num,-1))
        x_sum = np.sum(X_Sum,1)
        data_fea.append(x_sum)
    fea_matrix = np.array(data_fea)
    mean_matrix = np.mean(fea_matrix,0)
    fea_matrix = fea_matrix - np.tile(mean_matrix,(fea_matrix.shape[0],1))
    vari = np.sum(fea_matrix**2,0)
    idx = np.argsort(-vari)
    vari_new = vari[idx]
    return idx






'''
if __name__ == "__main__":
    dataset = 'oxford5k'
    fea_name = 'pool5'

    idx = selec_chan(dataset,fea_name)
'''