# -*- coding: utf-8 -*-
"""
Module contains 5 algorithms to detect small target in cluttered background
"""
import scipy.signal as sig
import numpy as np
import cv2
from tensorflow.image import non_max_suppression
import tensorflow as tf
import pywt


def find_clusters(fmap, bin_thresh = "adaptive"):
    """
    Find all clusters of white pixels in feature map fmap and
    perform non maximum supression
    bin_thresh - threshold mode for segmentation (if "adaptive" -> calculated)
    Returns ndarrays with shape [num_boxes, box_coord]
    box coordinates format : (x1, y1, x2, y2)
    """
    H, W = fmap.shape # height and width of feature map
    if bin_thresh == "adaptive":
        Km = (fmap.max() - fmap.mean()) / np.std(fmap)
        Kmean = (fmap[fmap > 0].mean() - fmap.mean()) / np.std(fmap)
        K = 0.5 * (Km + Kmean)
        print(K)
        bin_thresh = fmap.mean() + K * np.std(fmap)
    else:
        bin_thresh = fmap.mean() + bin_thresh * np.std(fmap)
    fmap = (fmap > bin_thresh)*255
    out = []
    for _ in range(10):
        for y in range(H - 1):
            for x in range(W - 1):
                if fmap[y, x] == 255 and fmap[y + 1, x + 1] == 255:
                    fmap[y + 1, x] = 255
                    fmap[y, x + 1] = 255
                if fmap[y, x + 1] == 255 and fmap[y + 1, x] == 255:
                    fmap[y, x] = 255
                    fmap[y + 1, x + 1] = 255
    for y in range(H):
        for x in range(W):
            if fmap[y,x] == 255:
                k = 1
                while y + k <= H - 1 and fmap[y + k, x] == 255:
                    k += 1
                d = 1
                while x + d <= W - 1 and fmap[y, x + d] == 255:
                    d += 1
                if k > 1 and d > 1 and y + k <= H - 1 and x + d <= W - 1:
                    fmap[y : y + k, x : x + d] = 0
                    out.append(np.array((x, y, x + d, y + k)))
                elif x > 0 and y > 0:
                    out.append(np.array((x - 1, y - 1, x + 1, y + 1)))

    clusters = np.array(out)
    if clusters.shape[0] > 1:
        clusters = tf.gather(clusters, 
                         non_max_suppression(clusters,
                                             np.ones(clusters.shape[0]), 25, 0.00000001))
        return clusters.numpy()
    return clusters

def fractal_clf(fmap, clst, thresh, max_obj = 1):
    """
    Perform classification of ROI clusters based on fractal dimension and
    autocorrelation values
    
    fmap - feature map
    
    clst - ROI clusters with shape [num_boxes, 4]
    box coordinates format : (x1, y1, x2, y2)
    
    thresh - classification threshold
    
    max_obj can be used to sort weight vector and get boxes with highest weight
    """
    feat = np.zeros((clst.shape[0], 2))
    # loop over ROI
    for i in range(clst.shape[0]):
        roi = fmap[clst[i,1] : clst[i,3], clst[i,0] : clst[i,2]]
        M, N = roi.shape
        a = min(M,N)
        sumfr = np.zeros((a))
        # compute fractal dimension based pixel covering
        for Q in range(a):
            tx = np.floor(M / (Q + 1)).astype("int32")
            ty = np.floor(N / (Q + 1)).astype("int32")
            s = 0 # sum
            for X in range(tx):
                for Y in range(ty):
                    m = np.min(roi[Q*X : Q*(X + 1) + 1, Q*Y : Q*(Y + 1) + 1])
                    s += np.ceil(m / (Q + 1))
            if s == 0:
                s = 1
            sumfr[Q] = s
        A11=a; A12=0; A21=0; A22=0; d1=0; d2=0;
        for iv in range(a):
            A12 += np.log(iv + 1)
            A22 += (np.log(iv + 1)) ** 2
            d1 += np.log(sumfr[iv])
            d2 += np.log(sumfr[iv]) * np.log(iv + 1);
        A21=A12; D=A11*A22-A12*A21;
        D2=A11*d2-A21*d1; m=D2/(D*1.4);
        feat[i,0] = np.abs(m)
        
        # compute autocorrelation of ROI and find max eugen value
        sig2 = np.var(roi)
        roi_r = np.roll(roi, 1, axis = 0)
        roi_c = np.roll(roi, 1, axis = 1)
        rho = np.corrcoef(
            np.concatenate((roi.flatten(), roi.flatten())),
            np.concatenate((roi_c.flatten(), roi_r.flatten()))
            )[0,1]
        rr, cc = np.meshgrid(np.arange(-a,a), np.arange(-a,a))
        r_x = sig2 * rho * np.sqrt(rr ** 2 + cc ** 2)
        feat[i, 1] = np.max(np.linalg.eigvals(r_x))
        
    feat[:, 0] = feat[:, 0] / feat[:, 0].max()
    feat[:, 1] = feat[:, 1] / feat[:, 1].max()
    
    w = 2 * feat[:,0] * feat[:,1] / (feat[:,0] + feat[:,1])
    #ind = np.logical_and(feat[:,0] >= 1.5, feat[:,1] >= 1000)
    #ind = np.flip(np.argsort(w))[:max_obj]
    ind = w > thresh
    return clst[ind, :].reshape(-1,4)

def multy_feature_clf(fmap, clst, thresh, max_obj = 1):
    """
    Perform Fuzzy Multy-Feature Decision Making from:
    Yang, D.; Bai, Z.; Zhang, J. Infrared Weak and Small Target Detection 
    Based on Top-Hat Filtering and Multi-Feature 
    Fuzzy Decision-Making. Electronics 2022, 11, 3549
    
    fmap - feature map
    
    clst - ROI clusters with shape [num_boxes, 4]
    box coordinates format : (x1, y1, x2, y2)
    
    thresh - classification threshold
    
    max_obj can be used to sort weight vector and get boxes with highest weight
    """
    n = clst.shape[0] # number of detections
    maxes = [fmap[clst[i,1] : clst[i,3], clst[i,0] : clst[i,2]].max() 
             for i in range(n)]
    
    def find_orientation_grads(roi):
        cy, cx = (roi.shape[0] - 1) // 2, (roi.shape[1] - 1) // 2
        og = roi[cy, -1] # right
        og += roi[cy, 0] # left
        og += roi[0, cx] # up
        og += roi[-1, cx] # down
        og += roi[0, 0] # up left
        og += roi[-1, -1] # down right
        og += roi[0, -1] # up right
        og += roi[-1, 0] # down left
        return 8 * roi[cy, cx] - og
    
    orient_grads = [find_orientation_grads(
        fmap[clst[i,1] : clst[i,3], clst[i,0] : clst[i,2]]
        ) for i in range(n)]
    
    def find_central_pixel_contrast(roi):
        cy, cx = (roi.shape[0] - 1) // 2, (roi.shape[1] - 1) // 2
        Ic = roi[cy, cx] # central pixel value
        mean_v = (roi.sum() - Ic) / (roi.shape[0] * roi.shape[1] - 1) 
        return (Ic ** 2) / (mean_v + 0.00001)
    
    c_pix_contrast = [find_central_pixel_contrast(
        fmap[clst[i,1] : clst[i,3], clst[i,0] : clst[i,2]]
        ) for i in range(n)]
    
    def find_region_grads(roi):
        cy, cx = (roi.shape[0] - 1) // 2, (roi.shape[1] - 1) // 2
        W1 = (np.triu(roi, 1).sum() - np.tril(roi, -1).sum()) / (
            np.triu(np.ones(roi.shape), 1).sum())
        W2 = (np.tril(np.fliplr(roi), -1).sum() - 
              np.triu(np.fliplr(roi), 1).sum()) / (
                  np.triu(np.ones(roi.shape), 1).sum())
        W3 = (roi[:, cx + 1 :].sum() - roi[:, : cx].sum()) / (
            np.ones((roi[:, cx + 1 :].shape)).sum())
        W4 = (roi[cy + 1 :, :].sum() - roi[: cy, :].sum()) / (
            np.ones((roi[cy + 1 :, :].shape)).sum())
        return 1 / (max(W1, W2, W3, W4) + 0.000001)
    
    region_grads = [find_region_grads(
        fmap[clst[i,1] : clst[i,3], clst[i,0] : clst[i,2]]
        ) for i in range(n)]
    
    # feature matrix R
    R = np.array([maxes, orient_grads, c_pix_contrast, region_grads])
    # return R
    # processing for obtain relation matrix U
    U = np.zeros(R.shape)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (R[i,j] - (1 / (j + 1)) * R[i, 0 : (j + 1)].sum()) > 0:
                U[i, j] = R[i, j] * np.exp(
                    (R[i,j] / ((1 / (j + 1)) * R[i, 0 : (j + 1)].sum()) - 1))
    # return U
    # processing for obtain weight vector A based on feature matrix R
    A = np.zeros(R.shape[0])
    for k in range(A.shape[0]):
        d = 0
        for i in range(R.shape[1]):
            for j in range(R.shape[1]):
                d += (R[k, i] - R[k, j]) ** 2
        A[k] = d / (R.shape[1] ** 2)
    # return A
    # apply weight vector A to relation matrix U for obtain fuzzy decision
    # probability matrix B
    B = np.matmul(A.reshape(1, R.shape[0]),U)
    #ind = np.flip(np.argsort(B))[0, :max_obj]
    ind = ((B / B.max()) > thresh).squeeze()
    return clst[ind, :].reshape(-1,4)
    
    

def min_loc_LoG(img, k_size = 9, sigma = 1.8):
    """
    Perform min-loc-LoG filtering of grayscale image img
    Sungho K. Min-local-LoG Filter for Detecting Small Targets in 
    Cluttered Background // Electronics Letters. 
    – 2011. – Vol. 47. – № 2. – P. 105-106. DOI: 10.1049/el.2010.2066.

    sigma - std of gaussian
    k_size - size of kernel
    """
    x = np.arange(k_size).reshape(1, k_size)
    y = np.arange(k_size).reshape(k_size, 1)
    # generate fE (positive X)
    fE = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fE[fE > 0] = fE[fE > 0] / fE[fE > 0].sum()
    fE[fE < 0] = fE[fE < 0] / (-fE[fE < 0].sum())
    # generate fS (positive Y)
    fS = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fS[fS > 0] = fS[fS > 0] / fS[fS > 0].sum()
    fS[fS < 0] = fS[fS < 0] / (-fS[fS < 0].sum())
    # generate fW
    x = - np.fliplr(x)
    fW = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fW[fW > 0] = fW[fW > 0] / fW[fW > 0].sum()
    fW[fW < 0] = fW[fW < 0] / (-fW[fW < 0].sum())
    # generate fN
    y = - np.flipud(y)
    fN = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fN[fN > 0] = fN[fN > 0] / fN[fN > 0].sum()
    fN[fN < 0] = fN[fN < 0] / (-fN[fN < 0].sum())
    # perform 2D convolution with kernels
    def move(img, x, y):
        move_matrix = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv2.warpAffine(img, move_matrix, dimensions)

    Ie = sig.convolve2d(move(img, 4, 0), fE, mode = "same")
    Is = sig.convolve2d(move(img, 0, 4), fS, mode = "same")
    Iw = sig.convolve2d(move(img, -4, 0), fW, mode = "same")
    In = sig.convolve2d(move(img, 0, -4), fN, mode = "same")
    f = np.dstack((Ie, Is, Iw, In))
    fmap = np.min(f, axis = 2)
    return fmap

def min_loc_LoG_clf(img, thresh, k_size = 9, sigma = 1.8):
    """
    Perform min-loc-LoG filtering of grayscale image img and
    classification with threshold segmentation
    
    Sungho K. Min-local-LoG Filter for Detecting Small Targets in 
    Cluttered Background // Electronics Letters. 
    – 2011. – Vol. 47. – № 2. – P. 105-106. DOI: 10.1049/el.2010.2066.
    
    thresh - threshold for fmap segmentation
    
    sigma - std of gaussian
    k_size - size of kernel
    """
    fmap = min_loc_LoG(img, k_size = 9, sigma = 1.8)
    return find_clusters(fmap, thresh)
    

def LCM(img, w_size = 15):
    """
    Perform LCM filtering of grayscale image img
    
    Chen C.L.P., Li H., Wei Y., Xia T., Tang Y.Y. 
    A Local Contrast Method for Small Infrared Target Detection 
    // IEEE Transactions on Geoscience and Remote Sensing. – 2014. 
    – Vol. 52. – № 1. – P. 574-581. DOI: 10.1109/TGRS.2013.2242477
    
    w_size - size of square sliding window
    """
    def roll(a,      # ND array
             k_size,      # rolling 2D window array
             dx=1,   # horizontal step, abscissa, number of columns
             dy=1):  # vertical step, ordinate, number of rows
        shape = a.shape[:-2] + \
                ((a.shape[-2] - k_size[-2]) // dy + 1,) + \
                ((a.shape[-1] - k_size[-1]) // dx + 1,) + \
                k_size  # sausage-like shape with 2D cross-section
        strides = a.strides[:-2] + \
                  (a.strides[-2] * dy,) + \
                  (a.strides[-1] * dx,) + \
                  a.strides[-2:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    patches = roll(np.pad(img, (w_size - 1) // 2),
                   (w_size, w_size)).reshape(-1, w_size, w_size)
    
    def LCM_processing(roi):
        d = w_size // 3
        L = roi[d : 2*d, d : 2*d].max()
        c1 = L**2 / roi[:d, :d].mean()
        c2 = L**2 / roi[:d, d : 2*d].mean()
        c3 = L**2 / roi[:d, 2*d:].mean()
        c4 = L**2 / roi[d : 2*d, :d].mean()
        c5 = L**2 / roi[d : 2*d, 2*d:].mean()
        c6 = L**2 / roi[2*d:, :d].mean()
        c7 = L**2 / roi[2*d:, d : 2*d].mean()
        c8 = L**2 / roi[2*d:, 2*d:].mean()
        if L == 0:
            return 0
        else:
            return min(c1, c2, c3, c4, c5, c6, c7, c8)
    
    out = np.array([LCM_processing(roi) for roi in patches])
    
    return out.reshape(img.shape[0], img.shape[1])

def LCM_clf(img, thresh, w_size = 15):
    """
    Perform LCM filtering of grayscale image img and
    classification with threshold segmentation
    
    Chen C.L.P., Li H., Wei Y., Xia T., Tang Y.Y. 
    A Local Contrast Method for Small Infrared Target Detection 
    // IEEE Transactions on Geoscience and Remote Sensing. – 2014. 
    – Vol. 52. – № 1. – P. 574-581. DOI: 10.1109/TGRS.2013.2242477
    
    thresh - threshold for fmap segmentation
    
    w_size - size of square sliding window
    """
    fmap = LCM(img, w_size)
    return find_clusters(fmap, thresh)

def ACSD(img, M = 5, N = 9, k = 0.15):
    """
    Perform ACSD filtration on image img
    
    Xie K., Fu K., Zhou T., Zhang J., Yang J., Wu Q. 
    Small Target Detection Based On Accumulated Center-Surround Difference Measure 
    // Infrared Physics & Technology. – 2014. – Vol. 67. – P. 229-236. 
    DOI: 10.1016/j.infrared.2014.07.006
    
    M - size of inner square window
    N - size of outer square window, so N > M
    k - coef in weight function
    """
    def roll(a,      # ND array
             k_size,      # rolling 2D window array
             dx=1,   # horizontal step, abscissa, number of columns
             dy=1):  # vertical step, ordinate, number of rows
        shape = a.shape[:-2] + \
                ((a.shape[-2] - k_size[-2]) // dy + 1,) + \
                ((a.shape[-1] - k_size[-1]) // dx + 1,) + \
                k_size  # sausage-like shape with 2D cross-section
        strides = a.strides[:-2] + \
                  (a.strides[-2] * dy,) + \
                  (a.strides[-1] * dx,) + \
                  a.strides[-2:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    patches = roll(np.pad(img.astype('float32'), (N - 1) // 2),
                   (N, N)).reshape(-1, N, N)
    
    def ACSD_processing(roi):
        c = (N - 1) // 2 # coordinate of central pixel
        I_c = roi[c, c] # intensity of central pixel
        
        # for vertical and horizontal directions
        ap_num = (N - M) // 2 # number of activ pixels 
        ap = np.arange(ap_num) + (M + 1)//2 # np array of active pixel coordinates
        w_func = 1 - np.exp(-k*(ap**2)) # weight function
        ACSD_0 = (np.absolute(roi[c, c + (M + 1)//2 :] - I_c) * w_func).sum() # hor right
        ACSD_180 = (np.absolute(np.flip(roi[c, : c - (M - 1)//2] - I_c)) * w_func).sum() # hor left
        ACSD_90 = (np.absolute(roi[c + (M + 1)//2 :, c] - I_c) * w_func).sum() # ver down
        ACSD_270 = (np.absolute(np.flip(roi[: c - (M - 1)//2, c] - I_c)) * w_func).sum() # ver up

        # for diagonal directions 45 and 225
        diag = roi.diagonal()
        ACSD_45 = (np.absolute(diag[c + (M + 1)//2 :] - I_c) * w_func).sum() # right down
        ACSD_225 = (np.absolute(np.flip(diag[: c - (M - 1)//2] - I_c)) * w_func).sum() # up left

        # for diagonal directions 135 and 315
        diag = np.fliplr(roi).diagonal()
        ACSD_135 = (np.absolute(diag[c + (M + 1)//2 :] - I_c) * w_func).sum() # down left
        ACSD_315 = (np.absolute(np.flip(diag[: c - (M - 1)//2] - I_c)) * w_func).sum() # up right
        return min(ACSD_0, ACSD_45, ACSD_90, ACSD_135, ACSD_180,
                   ACSD_225, ACSD_270, ACSD_315)
    
    out = np.array([ACSD_processing(roi) for roi in patches])
    return out.reshape(img.shape[0], img.shape[1])

def ACSD_clf(img, thresh, M = 5, N = 9, k = 0.15):
    """
    Perform ACSD filtration on image img
    
    Xie K., Fu K., Zhou T., Zhang J., Yang J., Wu Q. 
    Small Target Detection Based On Accumulated Center-Surround Difference Measure 
    // Infrared Physics & Technology. – 2014. – Vol. 67. – P. 229-236. 
    DOI: 10.1016/j.infrared.2014.07.006
    
    M - size of inner square window
    N - size of outer square window, so N > M
    k - coef in weight function
    """
    fmap = ACSD(img, M, N, k)
    return find_clusters(fmap, thresh)

def wavelet(img):
    """
    Perform wavelet feature map extraction from grayscale image img
    column-wise and row-wise with mexican hat wavelet
    """
    H, W = img.shape
    rows = []
    for y in range(H):
        rows.append(pywt.cwt(img[y, :], 1, 'mexh')[0])
    rows = np.abs(np.array(rows).squeeze(1))
    cols = []
    for x in range(W):
        cols.append(pywt.cwt(img[:, x], 1, 'mexh')[0])
    cols = np.abs(np.array(cols).squeeze(1).T)
    return np.amin(np.dstack((rows, cols)), axis = 2)

def top_hat(img, k_size = (5,5)):
    """
    Perform top-hat feature map extraction from grayscale image img
    with structuring element having size of k_size
    """
    se = np.ones(k_size).astype('uint8')
    fmap = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
    return fmap

def top_hat_clf(img, thresh):
    """
    Perform Fuzzy Multy-Feature Decision Making from:
    
    Yang, D.; Bai, Z.; Zhang, J. Infrared Weak and Small Target Detection 
    Based on Top-Hat Filtering and Multi-Feature 
    Fuzzy Decision-Making. Electronics 2022, 11, 3549
    
    1) Get feature map fmap with top-hat filtering
    2) Segment fmap to find ROI clusters with find_clusters
    3) Classify clusters with multy_feature_clf according to thresh (0...1)
    """
    fmap = top_hat(img)
    clusters = find_clusters(fmap, "adaptive")
    clf = multy_feature_clf(fmap, clusters, thresh)
    return clf

def wavelet_clf(img, thresh):
    """
    Perform wavelet feature map extraction and classification based on 
    fractal dimension and autocorrelation value
    
    1) Get feature map fmap with wavelet filtering
    2) Segment fmap to find ROI clusters with find_clusters
    3) Classify clusters with fractal_clf according to thresh (0...1)
                      
    Segmentation threshold in find_clusters can be variable, but set to 13 
    according to research
    """
    fmap = wavelet(img)
    clusters = find_clusters(fmap, 13)
    clf = fractal_clf(fmap, clusters, thresh)
    return clf
