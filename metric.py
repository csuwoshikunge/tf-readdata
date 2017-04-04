from PIL import Image
import numpy as np
import skimage.io as skio
from scipy.misc import imresize as sp_imresize
def error_metrics(depth_pred, depth_gt, mask=None):
    '''
    :param depth_pred: predicted image depth, 2d array
    :param depth_gt: the ground truth image depth of images, 2d darray
    :param mask:???
    :return: rel error, rms error, lg10 error
    '''
    if depth_pred.shape!=depth_gt.shape:
        depth_pred = sp_imresize(depth_pred,[depth_gt.shape[0],depth_gt.shape[1]])

    n_pxls = depth_gt.shape[0]*depth_gt.shape[1]
    # Mean Absolute Relative Error
    rel = np.divide(np.abs(depth_gt-depth_pred),depth_gt) # compute errors
    #rel(~mask) = 0                        # mask out invalid ground truth pixels
    rel = np.sum(rel) / n_pxls

    # Root Mean Squared Error
    rms = np.square(depth_gt-depth_pred)
    #rms(~mask) = 0
    rms = np.sqrt(np.sum(rms) / n_pxls)

    # LOG10 Error
    lg10 = np.abs(np.log10(depth_gt) - np.log10(depth_pred));
    #lg10(~mask) = 0
    lg10 = np.sum(lg10) / n_pxls

    return rel,rms,lg10