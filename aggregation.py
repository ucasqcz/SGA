import numpy as np
import os
import scipy
import sys
from sklearn.preprocessing import normalize as sknorm

def aggregate(x,idx,method = '',prms = None):
    if prms is None:
        prms = {}
        prms['constrant'] = 10**-3
        prms['constrantGaussian'] = 10**-16
        prms['constrantdet'] = 10**-12
        prms['alpha'] = 0.75
        prms['embedding_beta'] = 0.3
    if method == 'raid_g':

        dim = x.shape[0]
        x = np.reshape(x,(dim,-1))

        ## feature selection
        x = x[idx,:]
        dim = x.shape[0]

        # root fea
        x = np.sign(x)*(np.abs(x)**0.5)
        true_mat = np.ones(shape = (dim+1,dim+1),dtype=np.bool)
        in_triu = np.triu(true_mat)
        regionD = np.ones(shape = (dim +1,dim +1))
        alpha = prms['alpha']
        gama = (1-alpha)/(2*alpha)
        b = prms['embedding_beta']
        b2 = b**2
        shift = prms['constrant'] * np.eye(dim)

        covd = np.cov(x) + shift

        u,s,v = np.linalg.svd(covd)
        diag_s = np.sign(s)*((gama**2 + np.abs(s/alpha))**0.55 - gama)
        covd = np.dot(np.dot(u,np.diag(diag_s)),np.transpose(u))
        regionD[0:dim,0:dim] = covd + b2*np.dot(np.mean(x,1),np.transpose(np.mean(x,1)))
        regionD[0:dim,dim] = b*np.mean(x,1)
        regionD[dim,0:dim] = b*np.mean(x,1)

        u,s,v = np.linalg.svd(regionD)
        s = s + prms['constrantGaussian']
        s = np.sign(s)*(np.abs(s)**0.9)
        regionD = np.dot(np.dot(u,np.diag(s)),np.transpose(u))
        y = regionD[in_triu]
        #y = sknorm(y,norm= 'l2')
        return y
    if method == 't_em':
        dim = x.shape[0]
        x = np.reshape(x,[dim,-1])
        prms['t_v'] = 3
        mu = np.zeros((dim,1),np.float)
        R = np.zeros((dim,dim),np.float)
        w = np.ones((dim,1),np.float)
        mu_new = np.dot(x,w) / np.sum(w)







