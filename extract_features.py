# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import caffe
import numpy as np
from PIL import Image
import glob
import scipy


###################################
# Feature Extraction
###################################

def load_img(path):
    """so
    Load the image at the provided path and normalize to RGB.

    :param str path:
        path to image file
    :returns Image:
        Image object
    """
    try:
        img = Image.open(path)
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        return rgb_img
    except:
        return None



def format_img_for_vgg(img):
    """
    Given an Image, convert to ndarray and preprocess for VGG.

    :param Image img:
        Image object
    :returns ndarray:
        3d tensor formatted for VGG
    """
    # Get pixel values
    d = np.array(img, dtype=np.float32)
    d = d[:,:,::-1]

    # Subtract mean pixel values of VGG training set
    d -= np.array((104.00698793,116.66876762,122.67891434))             # B G R

    return d.transpose((2,0,1))


def extract_raw_features(net, layer, d):
    """
    Extract raw features for a single image.
    """
    # Shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *d.shape)
    net.blobs['data'].data[...] = d
    # run net and take argmax for prediction
    net.forward()
    return net.blobs[layer].data[0]


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--datafold',dest='datafold',default='/home1/qcz/DataSet/')
    parser.add_argument('--dataset',dest='dataset',default='oxford5k')
    #parser.add_argument('--images', dest='images', type=str, nargs='+', default='/home1/qcz/DataSet/oxford5k/images/*.jpg', required=False, help='glob pattern to image data')
    parser.add_argument('--layer', dest='layer', type=str, default='pool5', help='model layer to extract')
    parser.add_argument('--prototxt', dest='prototxt', type=str, default='vgg/VGG_ILSVRC_16_pool5.prototxt', help='path to prototxt')
    parser.add_argument('--caffemodel', dest='caffemodel', type=str, default='vgg/VGG_ILSVRC_16_layers.caffemodel', help='path to model params')
    # parser.add_argument('--out', dest='out', type=str, default='paris6k/res5c', help='path to save output')
    parser.add_argument('--gpuid',dest='gpuid',type=int, default=0)
    args = parser.parse_args()
    out = os.path.join(args.dataset,args.layer)

    args.gpuid = 1
    caffe.set_mode_gpu()
    caffe.set_device(args.gpuid)

    #net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net = caffe.Net(args.prototxt,caffe.TEST )
    net.copy_from(args.caffemodel)

    if not os.path.exists(out):
        os.makedirs(out)
    images = os.path.join(args.datafold,args.dataset,'images/*.jpg')

    imgList = glob.glob(images)
    idx = 0


    for path in imgList:
        img = load_img(path)
        print "%s %d" % (path,idx)
        idx += 1


        # Skip if the image failed to load
        if img is None:
            print path
            continue

        d = format_img_for_vgg(img)
        X = extract_raw_features(net, args.layer, d)

        filename = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(out, filename), X)
