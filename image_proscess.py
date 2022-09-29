import numpy as np
import scipy.io

def im2mat(img):
    r""" 

    Vectorize image img 

    Parameters
    ----------
    img : array-like, shape (n, n, nb_canaux)

    Returns
    -------
    r :  array-like, shape (nb_canaux, N)
    """
    nb_canaux = img.shape[2]
    N = (img.shape[0])**2
    r = np.zeros((nb_canaux, N))
    for c in range(nb_canaux):
        r[c, :] = img[:, :, c].reshape((1, N))
    return r


def mat2im(F):
    r""" 

    DeVectorize image img 

    Parameters
    ----------
    F : array-like, shape (nb_canaux, N)

    Returns
    -------
    img_f : array-like, shape (n, n, nb_canaux)
    """
    n = round(np.sqrt(F.shape[1]))
    img_f = np.zeros((n, n, F.shape[0]))
    for c in range(F.shape[0]):
        img_f[:, :, c] = F[c, :].reshape(n, n)
    return img_f


def pos_xy(n):
    r""" 

    Create an array with position x and y of each pixel on the 2D grid
    Parameters
    ----------
    n : int
        num of pixels in each direction of the image x,y

    Returns
    -------
    pos : array-like, shape (2, n*n)
          spectral bands with the position of each pixel on the x,y 2D-grid
    """
    pos = np.zeros((2, n*n))
    for j in range(n*n):
        pos[:, j] = np.array([j//n, j % n])
    return pos


def add_posvect(V):
    r""" 

    Add position vector from pos_xy(n) to the image V

    Parameters
    ----------
    n : int
        num of pixels in each direction of the image x,y

    Returns
    -------
    pos : array-like, shape (2, n*n)
          spectral bands with the position of each pixel on the x,y 2D-grid
    """
    N = V.shape[1]
    n = round(np.sqrt(N))
    return np.vstack((V, pos_xy(n)))


def interpo(Iv, d):
    r""" 

    Interpolate image Iv building image Iint

    Parameters
    ----------
    Iv : array-like, shape (lh,ph)
         image to interpolate
    d : int 
        squared-root of interpolation ration d^2=pm/ph
    Returns
    -------
    Iint : array-like, shape (lh,pm)
           interpolated image
    """
    Im = mat2im(Iv)
    U = scipy.ndimage.zoom(Im, [d, d, 1])
    Iint = im2mat(U)
    return Iint

def torgb(im):
    r""" 

    Extract RGB image from HS image for example image

    Parameters
    ----------
    im : array-like, shape (l,p)
         HS image

    Returns
    -------
    Irgb : array-like, shape (l,p)
           RGB image
    """
    r = mat2im(im[26,:].reshape(1,im.shape[1]))
    g = mat2im(im[19,:].reshape(1,im.shape[1]))
    b = mat2im(im[10,:].reshape(1,im.shape[1]))
    r = r / np.max(r)
    g = g / np.max(g)
    b = b / np.max(b)
    Irgb = np.dstack((r,g,b))
    return Irgb