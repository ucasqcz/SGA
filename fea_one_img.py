import caffe
import os
from PIL import Image
import numpy as np
import scipy.io as scio

from extract_features import load_img,format_img_for_vgg,extract_raw_features

fold = '/home1/qcz/DataSet'
dataset = 'oxford5k'
name = 'ashmolean_000283.jpg'

imgPath = os.path.join(fold,dataset,'images',name)

prototxt = 'vgg/VGG_ILSVRC_16_pool5.prototxt'
caffemodel = 'vgg/VGG_ILSVRC_16_layers.caffemodel'
feaName = 'pool5'

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(prototxt,caffe.TEST)
net.copy_from(caffemodel)

im = load_img(imgPath)
im = format_img_for_vgg(im)

fea = extract_raw_features(net,feaName,im)
scio.savemat('fea.mat',{'fea':fea})

print 'fea'

