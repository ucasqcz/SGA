import os
import sys
import numpy as np
import glob
from math import *
from sklearn.decomposition import pca
from aggregation import aggregate
from crow import run_feature_processing_pipeline
from evaluate import get_ap
import time
import pickle
from sklearn.externals import joblib
import threading
from selec_chan import selec_chan


def aggregate_and_save(src_fold,method,dst_fold,fea_idx,total_order = 0,order = 0):
    if not os.path.exists(dst_fold):
        os.mkdir(dst_fold)
    src_fold = os.path.join(src_fold,'*.npy')

    fea_list = glob.glob(src_fold)
    if not total_order == 0 and not order == 0:
        length = len(fea_list)
        step = int(ceil( float(length) / total_order))
        fea_list = fea_list[(order - 1)*step:min(order*step,length)]
    idx = 0
    for path in fea_list:
        fold,name = os.path.split(path)
        save_path = os.path.join(dst_fold, name)
        if os.path.exists(save_path):
            idx += 1
            if not idx % 100:
                sys.stdout.write('\rEncoding %d th img: %s' % (idx, name))
                sys.stdout.flush()
            continue
        X = np.load(path)
        y = aggregate(X,fea_idx,method)

        np.save(save_path,y)
        idx += 1
        if not idx % 100:
            sys.stdout.write('\rEncoding %d th img: %s' % (idx, name))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()



def load_for_pca(src_fold,copy = False,pca_params = {}):
    src_fea = []
    names = []
    src_fold = os.path.join(src_fold, '*.npy')
    file_list = glob.glob(src_fold)
    idx = 0
    for path in file_list:
        X = np.load(path)
        #X = np.reshape(X,(-1,1))
        fold,name = os.path.split(path)
        name = os.path.splitext(name)[0]
        src_fea.append(X)
        names.append(name)
        idx += 1
        if not idx % 100:
            sys.stdout.write('\rLoading %d th img: %s' % (idx, name))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    st_time = time.time()
    if pca_params.has_key('pca'):
        d = pca_params['d']
        whiten = pca_params['whiten']
        features,params = run_feature_processing_pipeline(src_fea,d,whiten,False,pca_params)
        params['d'] = d
        params['whiten'] = whiten
    else:
        d = pca_params['d']
        whiten = pca_params['whiten']
        features, params = run_feature_processing_pipeline(src_fea, d, whiten, False)
        params['d'] = d
        params['whiten'] = whiten
    en_time = time.time()
    print('pca time is: %f' % (en_time - st_time))

    return features,params,names

def aggregate_and_save_mt(src_fold,method,dst_fold,th_num = 10):
    threads = []
    st_time = time.time()
    for i in np.arange(th_num):
        t = threading.Thread(target=aggregate_and_save,
                             args=(src_fold, method, dst_fold, th_num, i + 1))
        threads.append(t)
    for i in np.arange(th_num):
        threads[i].start()
    for i in np.arange(th_num):
        threads[i].join()
    en_time = time.time()
    print('encoding time is : %f' % (en_time - st_time))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--method', dest='method', type=str, default='raid_g',
                        help='weighting to apply for feature aggregation')
    parser.add_argument('--trainset', dest='trainset', type=str, default='paris6k')
    parser.add_argument('--testset', dest='testset', type=str, default='oxford5k')
    parser.add_argument('--datafold', dest='datafold', type=str, default='/home1/qcz/DataSet',
                        help='src fold of all datasets')
    parser.add_argument('--layer', dest='layer', type=str, default='pool5')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='groundtruth/',
                        help='directory containing groundtruth files')
    parser.add_argument('--d', dest='d', type=int, default=512, help='dimension of final feature')
    parser.add_argument('--out', dest='out', type=str, default=None, help='optional path to save ranked output')
    parser.add_argument('--tr',dest='total_order', type=int, default=0, help='total part num')
    parser.add_argument('--idx',dest='order', type=int, default=0,help='part index')
    parser.add_argument('--sd',dest='sd',type=int,default=150,help='feature selection by variance')

    args = parser.parse_args()

    ## feature selection
    idx = selec_chan(args.testset,args.layer)
    idx = idx[0:args.sd]


    ## coding train data
    th_num = 20
    src_train_fold = os.path.join(args.trainset,args.layer)
    dst_train_fold = os.path.join(args.trainset,args.layer + '_' + args.method + '_sd_' + '%d' % args.sd)
    aggregate_and_save(src_train_fold,args.method,dst_train_fold,idx,args.total_order,args.order)
    # aggregate_and_save_mt(src_train_fold,args.method,dst_train_fold,th_num)
    #multi threading
    ## coding test data
    src_test_fold = os.path.join(args.testset,args.layer)
    dst_test_fold = os.path.join(args.testset,args.layer + '_' + args.method + '_sd_' + '%d' % args.sd)
    aggregate_and_save(src_test_fold,args.method,dst_test_fold,idx,args.total_order,args.order)
    # aggregate_and_save_mt(src_test_fold,args.method,dst_test_fold,th_num)
    ## pca

    params = {}
    params['d'] = args.d
    params['whiten'] = True
    params_path = os.path.join(args.trainset,('%s_%s_sd_%d_pca_%d.pkl' % (args.layer,args.method,args.sd,args.d)))
    if os.path.exists(params_path):
        params = joblib.load(params_path)
    else:
        _,params,_ = load_for_pca(dst_train_fold,False,params)
        joblib.dump(params,params_path)
    #del train_fea
    test_fea,params,test_names = load_for_pca(dst_test_fold,False,params)
    ## query
    src_query_fold = os.path.join(args.testset,args.layer + '_queries')
    dst_query_fold = os.path.join(args.testset,args.layer + '_' + args.method + '_queries'+ '_sd_' + '%d' % args.sd)
    # coding query
    aggregate_and_save(src_query_fold,args.method,dst_query_fold,idx)
    query_fea,params,query_names = load_for_pca(dst_query_fold,False,params)
    ## eval
    groundtruth = os.path.join(args.datafold,args.testset,args.groundtruth)

    length = len(query_names)
    map = []
    for i in np.arange(length):
        q = query_fea[i,:]
        query_name = query_names[i]
        ap = get_ap(q,test_fea,query_name,test_names,groundtruth,None)
        map.append(ap)
    print 'fea selection: %d--- map :%f' % (args.sd,np.mean(map))








