import numpy as np
import image_proscess
import scipy.io
import scipy.ndimage
import math

def S_spat(img,d):
    r"""
    Spatial degradation operator S : 
    2D-subsampling : take 1/d² pixel of the image

    Parameters
    ----------
    img : array-like, shape (lh,pm)
          image to subsample 
    d : squared-root of subsampling ratio

    Returns
    -------
    Is : array-like, shape (lh, ph)
         subsampled image
    """
    (lh,pm) = img.shape
    n = round(np.sqrt(pm))

    Ir = image_proscess.mat2im(img)
    Ig = np.zeros((round(n/d),round(n/d),lh))
    for c in range(lh):
        Ig[:,:,c] = Ir[:,:,c][0:-1:d,0:-1:d]
    Is = image_proscess.im2mat(Ig) 
    return Is

def ST_spat(img,d):
    r"""
    Adjoint of the Spatial degradation operator S
    Reshape image by adding zeros in the image

    Parameters
    ----------
    img : array-like, shape (lh,pm)
          image to which is apply the adjoint of S
    d : int
        squared-root of subsampling ratio

    Returns
    -------
    Is : array-like, shape (lh, ph)
         output image
    """
    (lh,ph) = img.shape
    nh = round(np.sqrt(ph))
    Ir = image_proscess.mat2im(img)
    Ig = np.zeros((nh*d,nh*d,lh))
    for c in range(lh):
        Ig[:,:,c][0:-1:d,0:-1:d] = Ir[:,:,c]
    Is = image_proscess.im2mat(Ig)
    return Is

def L(Y,lm):
    r"""
    Spectral degradation operator L : Averaging filter over spectral bands 

    Parameters
    ----------
    Y : array-like, shape (lh, p)
        image to which is apply L
    lm : int
         number of spectral bands in the output image
    Returns
    -------
    Ys : array-like, shape (lm, p)
         output image
    """
    (lh,p) = Y.shape
    ds = lh // lm
    Ys = np.zeros((lm,p))
    for k in range(lm):
        Ys[k,:] = np.mean(Y[k*ds:(k+1)*ds,:],0)
    return Ys

def LT(Y,lh):
    r"""
    Adjoint of the Spectral degradation operator L
    Reshape image by adding spectral bands with zeros

    Parameters
    ----------
    img : array-like, shape (l,p)
          image to which is apply the adjoint of L
    lh : int
         number of spectral bands in the output image

    Returns
    -------
    Is : array-like, shape (lh, p)
         output image
    """
    (l,p) = Y.shape
    ds = math.ceil(lh/l)
    Ys = np.zeros((lh,p))
    for k in range(lh):
        Ys[k,:] = (1/ds)*Y[k//ds,:]
    return Ys

def B(img,sigma=1):
    r"""
    Spatial gaussian bluur operator B
    Apply a gaussian bluur to the image by 2D convolution 

    Parameters
    ----------
    img : array-like, shape (l,p)
          image to which is apply B
    sigma : standard deviation 

    Returns
    -------
    Ib : array-like, shape (l, p)
         output image
    """
    (l,p) = img.shape
    n = round(np.sqrt(p))
    I = image_proscess.mat2im(img)
    If = np.zeros((n,n,l))
    for c in range(l):
        If[:,:,c] = scipy.ndimage.gaussian_filter(I[:,:,c],sigma)
    Ib = image_proscess.im2mat(If)
    return Ib