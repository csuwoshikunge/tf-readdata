from PIL import Image
import numpy as np
import skimage.io as skio
from scipy.misc import imresize
import cv2

def error_metrics(depth_pred, depth_gt, mask=None):
    '''
    :param depth_pred: predicted image depth, 2d array
    :param depth_gt: the ground truth image depth of images, 2d darray
    :param mask:???
    :return: rel error, rms error, lg10 error
    '''
    if depth_pred.shape!=depth_gt.shape:
        #depth_pred = imresize(depth_pred,(depth_gt.shape[0],depth_gt.shape[1]))
        #depth_pred = cv2.resize(depth_pred,(depth_gt.shape[0],depth_gt.shape[1]))
        depth_pred = cv2.resize(depth_pred,(depth_gt.shape[1],depth_gt.shape[0])) #resize(src,(width,height))

    n_pxls = depth_gt.shape[0]*depth_gt.shape[1]
    print("n_pxls:",n_pxls)
    # Mean Absolute Relative Error
    rel = np.divide(np.abs(depth_gt-depth_pred),depth_gt) # compute errors
    #rel(~mask) = 0                        # mask out invalid ground truth pixels
    rel = np.sum(rel) / n_pxls
    # other ways compute rel
    #rel = np.divide(np.abs(depth_gt-depth_pred),depth_gt)/n_pxls
    #rel = np.sum(rel)


    # Root Mean Squared Error
    rms = np.square(depth_gt-depth_pred)
    #rms(~mask) = 0
    rms = np.sqrt(np.sum(rms) / n_pxls)
    #other ways compute rms
    #rms = np.square(np.abs(depth_gt-depth_pred)/np.sqrt(n_pxls))
    #rms = np.sqrt(np.sum(rms))

    # LOG10 Error
    lg10 = np.abs(np.log10(depth_gt) - np.log10(depth_pred))
    #lg10(~mask) = 0
    lg10 = np.sum(lg10) / n_pxls
    #other ways computing lg10 error
    #lg10 = np.abs(np.log10(depth_gt)/n_pxls - np.log10(depth_pred)/n_pxls)
    #lg10 = np.sum(lg10)


    return rel,rms,lg10